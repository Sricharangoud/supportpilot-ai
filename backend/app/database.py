"""SQLAlchemy models and database session management for tickets and request logs."""
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Ticket(Base):
    """Escalated customer support ticket awaiting human follow-up."""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    intent = Column(String(50), nullable=False)
    reason = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(32), default="open", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


class ConversationLog(Base):
    """Audit log of every /chat request, retrieval sources, decision, and latency."""

    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    intent = Column(String(50), nullable=False)
    sources_json = Column(Text, nullable=False, default="[]")
    confidence = Column(Float, nullable=False)
    decision = Column(String(32), nullable=False)
    answer = Column(Text, nullable=False)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


def create_db_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def init_db(database_url: str):
    engine = create_db_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
