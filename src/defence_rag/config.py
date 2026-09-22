from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str | None = None
    portkey_api_key: str | None = None
    portkey_provider: str = "@openrouter"
    portkey_base_url: str = "https://api.portkey.ai/v1"

    langsmith_api_key: str | None = None
    langsmith_tracing: bool = True
    langsmith_project: str = "defence-rag"

    primary_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    fallback_models: str = (
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,"
        "google/gemma-4-31b-it:free,qwen/qwen3.8-27b:free"
    )
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embeddings_offline: bool = False

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "defence_kb"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    max_iterations: int = 3

    data_dir: Path = ROOT_DIR / "data" / "raw"
    eval_dir: Path = ROOT_DIR / "data" / "eval"
    checkpoint_path: Path = ROOT_DIR / "data" / "runtime" / "checkpoints.db"

    api_key: str | None = None

    def apply_observability_env(self) -> None:
        if self.langsmith_api_key:
            os.environ.setdefault("LANGSMITH_API_KEY", self.langsmith_api_key)
        os.environ.setdefault("LANGSMITH_TRACING", str(self.langsmith_tracing).lower())
        os.environ.setdefault("LANGSMITH_PROJECT", self.langsmith_project)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", str(self.langsmith_tracing).lower())

    @property
    def fallback_model_list(self) -> list[str]:
        return [m.strip() for m in self.fallback_models.split(",") if m.strip()]

    @property
    def has_credentials(self) -> bool:
        return bool(self.openrouter_api_key and self.portkey_api_key)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.apply_observability_env()
    return settings