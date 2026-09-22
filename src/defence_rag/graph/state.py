from typing import TypedDict

from langchain_core.documents import Document


class RagState(TypedDict, total=False):
    query: str
    rewritten_query: str
    documents: list[Document]
    context: str
    generation: str
    loop_count: int
    route: str
    guard_blocked: bool
    guard_reasons: list[str]
    no_relevant: bool
    guided: bool
    output_guard_fail: bool
    citations: list[dict]
    supported: bool
    useful: bool
    chat_history: list[dict]


def initial_state(query: str) -> RagState:
    return {
        "query": query,
        "rewritten_query": query,
        "documents": [],
        "context": "",
        "generation": "",
        "loop_count": 0,
        "route": "vectorstore",
        "guard_blocked": False,
        "guard_reasons": [],
        "no_relevant": False,
        "guided": False,
        "output_guard_fail": False,
        "citations": [],
    }