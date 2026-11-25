"""FastAPI entrypoint for SupportPilot AI."""
from contextlib import asynccontextmanager
import json
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.agent import SupportPilotAgent
from app.config import Settings, get_settings
from app.database import ConversationLog, Ticket, init_db
from app.llm import get_llm_provider
from app.rag import VectorStore
from app.schemas import (
    AdminMetricsOut,
    ChatRequest,
    ChatResponse,
    ConversationLogOut,
    HealthOut,
    IngestResponse,
    TicketOut,
)

settings: Settings = get_settings()
SessionLocal = init_db(settings.database_url)
vector_store = VectorStore(
    index_path=settings.faiss_index_path,
    embedding_model=settings.embedding_model,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load existing FAISS index on startup, or build it automatically from KB_DIR
    loaded = vector_store.load()
    if not loaded and Path(settings.kb_dir).exists():
        vector_store.ingest_directory(
            settings.kb_dir,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    yield


app = FastAPI(
    title="SupportPilot AI",
    description="RAG-powered customer support agent with intent routing, guardrails, and escalation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(
        status="ok",
        index_loaded=vector_store.is_loaded,
        indexed_chunks=len(vector_store.chunks),
        llm_provider=settings.llm_provider,
        llm_configured=bool(settings.llm_api_key),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        docs_count, chunks_count = vector_store.ingest_directory(
            settings.kb_dir,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        return IngestResponse(docs_indexed=docs_count, chunks_indexed=chunks_count)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    if not vector_store.is_loaded:
        raise HTTPException(status_code=503, detail="Knowledge base index not loaded.")
    llm = get_llm_provider(settings)
    agent = SupportPilotAgent(llm=llm, vector_store=vector_store, settings=settings)
    try:
        return agent.handle_message(req.message, db)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/admin/tickets", response_model=list[TicketOut])
def list_tickets(db: Session = Depends(get_db)):
    return db.query(Ticket).order_by(Ticket.created_at.desc()).limit(100).all()


@app.get("/admin/conversations", response_model=list[ConversationLogOut])
def list_conversations(db: Session = Depends(get_db)):
    logs = db.query(ConversationLog).order_by(ConversationLog.created_at.desc()).limit(100).all()
    return [
        ConversationLogOut(
            id=log.id,
            user_message=log.user_message,
            intent=log.intent,
            sources=json.loads(log.sources_json or "[]"),
            confidence=log.confidence,
            decision=log.decision,
            answer=log.answer,
            latency_ms=log.latency_ms,
            created_at=log.created_at,
        )
        for log in logs
    ]


@app.get("/admin/metrics", response_model=AdminMetricsOut)
def get_metrics(db: Session = Depends(get_db)):
    logs = db.query(ConversationLog).all()
    total = len(logs)
    if total == 0:
        return AdminMetricsOut(
            total_requests=0,
            auto_resolved=0,
            escalated=0,
            refused=0,
            auto_resolve_rate=0.0,
            escalation_rate=0.0,
            avg_latency_ms=0.0,
        )
    auto_resolved = sum(1 for l in logs if l.decision == "auto_reply")
    escalated = sum(1 for l in logs if l.decision == "escalate")
    refused = sum(1 for l in logs if l.decision == "refuse")
    avg_latency = sum(l.latency_ms for l in logs) / total
    return AdminMetricsOut(
        total_requests=total,
        auto_resolved=auto_resolved,
        escalated=escalated,
        refused=refused,
        auto_resolve_rate=round(auto_resolved / total, 4),
        escalation_rate=round(escalated / total, 4),
        avg_latency_ms=round(avg_latency, 2),
    )
