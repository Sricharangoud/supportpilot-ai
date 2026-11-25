"""Pydantic schemas for API requests, responses, and structured LLM outputs."""
from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    billing = "billing"
    shipping = "shipping"
    returns = "returns"
    account = "account"
    technical = "technical"
    other = "other"


class IntentResult(BaseModel):
    """Validated JSON output from the intent classification LLM call."""

    intent: IntentType = IntentType.other
    is_sensitive: bool = False
    sensitivity_reason: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class SourceCitation(BaseModel):
    doc_name: str
    chunk_id: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    decision: Literal["auto_reply", "escalate", "refuse"]
    answer: str
    intent: IntentType
    confidence: float
    sources: list[SourceCitation]
    ticket_id: Optional[int] = None
    latency_ms: float


class IngestResponse(BaseModel):
    docs_indexed: int
    chunks_indexed: int


class TicketOut(BaseModel):
    id: int
    user_message: str
    intent: str
    reason: str
    confidence: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationLogOut(BaseModel):
    id: int
    user_message: str
    intent: str
    sources: list[dict]
    confidence: float
    decision: str
    answer: str
    latency_ms: float
    created_at: datetime


class AdminMetricsOut(BaseModel):
    total_requests: int
    auto_resolved: int
    escalated: int
    refused: int
    auto_resolve_rate: float
    escalation_rate: float
    avg_latency_ms: float


class HealthOut(BaseModel):
    status: str
    index_loaded: bool
    indexed_chunks: int
    llm_provider: str
    llm_configured: bool
