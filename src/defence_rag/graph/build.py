import os

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from defence_rag.config import Settings, get_settings
from defence_rag.guards import DefenceGuards, DefenceGuards as _G  # noqa: F401
from defence_rag.graph.edges import build_edges
from defence_rag.graph.nodes import RagNodes
from defence_rag.graph.state import RagState, initial_state
from defence_rag.vectorstore import DefenceVectorStore


def build_graph(
    llm,
    store: DefenceVectorStore,
    guards: DefenceGuards | None = None,
    settings: Settings | None = None,
    embeddings=None,
):
    settings = settings or get_settings()
    guards = guards or DefenceGuards()
    nodes = RagNodes(llm, store, guards, settings, embeddings)
    e_in, e_router, e_grade, e_hall, e_answer = build_edges(settings)

    graph = StateGraph(RagState)
    graph.add_node("input_guard", nodes.input_guard)
    graph.add_node("router", nodes.router)
    graph.add_node("off_topic", nodes.off_topic)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("generate", nodes.generate)
    graph.add_node("hallucination_check", nodes.hallucination_check)
    graph.add_node("answer_check", nodes.answer_check)
    graph.add_node("no_answer", nodes.no_answer)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", e_in, {"router": "router", "END": END})
    graph.add_conditional_edges(
        "router", e_router, {"off_topic": "off_topic", "rewrite": "rewrite"}
    )
    graph.add_edge("off_topic", "finalize")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        e_grade,
        {"no_answer": "no_answer", "rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("no_answer", "finalize")
    graph.add_edge("generate", "hallucination_check")
    graph.add_conditional_edges(
        "hallucination_check", e_hall, {"answer_check": "answer_check", "finalize": "finalize"}
    )
    graph.add_conditional_edges(
        "answer_check", e_answer, {"rewrite": "rewrite", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def build_checkpointer():
    checkpoint_dir = get_settings().checkpoint_path
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(checkpoint_dir))


def invoke_query(compiled, query: str, session_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": session_id or "default"}}
    config["recursion_limit"] = 8 + get_settings().max_iterations * 5
    result = compiled.invoke(initial_state(query), config)
    return result