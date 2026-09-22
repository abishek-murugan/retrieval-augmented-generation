from functools import lru_cache
import os
from typing import Iterable

import openai
from langchain_core.exceptions import (
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders

from defence_rag.config import Settings, get_settings

_FALLBACK_HANDLED: tuple[type[BaseException], ...] = (
    ModelRateLimitError,
    ModelError,
    ModelConnectionError,
    ModelTimeoutError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.APIStatusError,
)


def _build_chat_model(model: str, settings: Settings, max_tokens: int | None = None) -> ChatOpenAI:
    headers = createHeaders(api_key=settings.portkey_api_key, provider=settings.portkey_provider)
    return ChatOpenAI(
        model=model,
        base_url=settings.portkey_base_url,
        api_key=settings.portkey_api_key or settings.openrouter_api_key or "noop",
        default_headers=headers,
        temperature=settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        timeout=120,
        max_retries=2,
        request_timeout=120,
    )


def build_chat_with_fallbacks(max_tokens: int | None = None) -> BaseChatModel:
    settings = get_settings()
    models = [settings.primary_model, *settings.fallback_model_list]
    primary = _build_chat_model(models[0], settings, max_tokens)
    fallbacks = [_build_chat_model(m, settings, max_tokens) for m in models[1:]]
    if not fallbacks:
        return primary
    return primary.with_fallbacks(fallbacks=fallbacks, exceptions_to_handle=_FALLBACK_HANDLED)


@lru_cache
def get_chat_model() -> BaseChatModel:
    return build_chat_with_fallbacks()


def build_embeddings():
    from langchain_community.embeddings import FastEmbedEmbeddings

    settings = get_settings()
    os.environ["HF_HUB_OFFLINE"] = "1" if settings.embeddings_offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if settings.embeddings_offline else "0"
    return FastEmbedEmbeddings(model_name=settings.embed_model)


class CachedEmbeddings:
    """Wraps fastembed with a disk cache, so CPU-heavy embedding runs once."""

    def __init__(self, cache_path=None):
        self._embeddings = None
        self._cache: dict[str, list[float]] = {}
        self._cache_path = cache_path or (get_settings().checkpoint_path.parent / "embed_cache.json")
        self._load()

    def _load(self):
        if self._cache_path.exists():
            import json

            try:
                self._cache = json.loads(self._cache_path.read_text())
            except Exception:  # noqa: BLE001
                self._cache = {}

    def _save(self):
        import json

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))

    def _model(self):
        if self._embeddings is None:
            self._embeddings = build_embeddings()
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [(i, t, k) for i, (t, k) in enumerate(zip(texts, keys)) if k not in self._cache]
        if missing:
            batch = [t for _, t, _ in missing]
            vectors = self._model().embed_documents(batch)
            for (i, _, k), vec in zip(missing, vectors):
                self._cache[k] = vec
            self._save()
        return [self._cache[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self._model().embed_query(text)


@lru_cache
def get_cached_embeddings() -> CachedEmbeddings:
    return CachedEmbeddings()


def truncate_and_chunk(text: str, max_len: int = 6000) -> Iterable[str]:
    if len(text) <= max_len:
        yield text
        return
    yield text[:max_len]


def describe_llm(model: BaseChatModel) -> str:
    return getattr(model, "model", type(model).__name__)