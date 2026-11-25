"""Application configuration loaded from environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent


class Settings(BaseSettings):
    """Centralized runtime settings for SupportPilot AI."""

    llm_api_key: str = ""
    llm_provider: str = "groq"
    llm_model: str = "qwen/qwen3.8-27b"

    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = str(_BACKEND_DIR / "data" / "faiss_index")
    kb_dir: str = str(_BACKEND_DIR / "data" / "kb")

    confidence_threshold: float = 0.55
    top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50

    database_url: str = f"sqlite:///{(_BACKEND_DIR / 'supportpilot.db').as_posix()}"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    return Settings()
