from typing import Callable

from defence_rag.config import Settings
from defence_rag.graph.state import RagState

Edge = Callable[[RagState], str]


def build_edges(settings: Settings) -> tuple[Edge, Edge, Edge, Edge, Edge]:
    def edge_input_guard(state: RagState) -> str:
        return "END" if state.get("guard_blocked") else "router"

    def edge_router(state: RagState) -> str:
        return "off_topic" if state.get("route") == "off_topic" else "rewrite"

    def edge_grade_documents(state: RagState) -> str:
        if state.get("no_relevant"):
            if (state.get("loop_count") or 0) >= settings.max_iterations:
                return "no_answer"
            return "rewrite"
        return "generate"

    def edge_hallucination(state: RagState) -> str:
        return "answer_check" if state.get("supported") else "finalize"

    def edge_answer(state: RagState) -> str:
        if not state.get("useful") and (state.get("loop_count") or 0) < settings.max_iterations:
            return "rewrite"
        return "finalize"

    return edge_input_guard, edge_router, edge_grade_documents, edge_hallucination, edge_answer