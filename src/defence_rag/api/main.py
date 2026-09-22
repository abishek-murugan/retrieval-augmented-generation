from __future__ import annotations

import time

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from defence_rag.config import get_settings
from defence_rag.llm import build_chat_with_fallbacks
from defence_rag.vectorstore import DefenceVectorStore


def create_app() -> FastAPI:
    app = FastAPI(title="Defence RAG", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: dict = {}

    @app.on_event("startup")
    def _startup() -> None:
        settings = get_settings()
        state["settings"] = settings
        state["store"] = DefenceVectorStore(settings)
        state["llm"] = build_chat_with_fallbacks()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        store = state.get("store")
        if store:
            store.close()

    class AskRequest(BaseModel):
        question: str

    class AskResponse(BaseModel):
        answer: str
        citations: list[dict] = []
        sources: list[str] = []
        route: str | None = None
        loop_count: int | None = None
        guard_blocked: bool = False
        output_guard_fail: bool = False
        latency_ms: float = 0.0

    @app.get("/health")
    def health() -> dict:
        store: DefenceVectorStore = state["store"]
        return {
            "status": "ok",
            "collection": store.collection,
            "points": store.count,
            "credentialed": state["settings"].has_credentials,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        from defence_rag.guards import DefenceGuards
        from defence_rag.graph.build import build_graph
        from defence_rag.graph.state import initial_state

        settings = state["settings"]
        guards = DefenceGuards()
        graph = build_graph(state["llm"], state["store"], guards, settings=settings)

        t0 = time.perf_counter()
        result = graph.invoke(initial_state(req.question), {"recursion_limit": 20})
        latency = (time.perf_counter() - t0) * 1000

        return AskResponse(
            answer=result.get("generation", ""),
            citations=result.get("citations", []),
            sources=sorted({c.get("source", "") for c in result.get("citations", [])}),
            route=result.get("route"),
            loop_count=result.get("loop_count"),
            guard_blocked=result.get("guard_blocked", False),
            output_guard_fail=result.get("output_guard_fail", False),
            latency_ms=round(latency, 1),
        )

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("defence_rag.api.main:app", host="0.0.0.0", port=8000, reload=False)