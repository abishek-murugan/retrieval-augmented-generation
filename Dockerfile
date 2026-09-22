FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY src ./src
COPY scripts ./scripts

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "defence_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]