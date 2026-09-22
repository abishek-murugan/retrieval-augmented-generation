from defence_rag.config import get_settings

_SYSTEM = """You are a precise military analyst assistant for a national security and defence
knowledge base. You answer questions STRICTLY from the provided context. Ground every claim
in the retrieved evidence and cite the source name and page number like (Source: <source>, p.<page>).

Rules:
- If the context does not contain the answer, say "I don't know" — never invent facts.
- Do not repeat classified-style caveats that are absent from the source text.
- Be concise, structured with short paragraphs or bullets."""
_H = get_settings()  # noqa: F841 - keep settings import side effects (env) explicit

ROUTER_SYSTEM = """You are a query router for a defence & national security RAG system.
Decide whether the user's question should be answered from the knowledge base (a corpus of
unclassified national security documents: NATO, US DoD, air force doctrine, defence policy)
or handled directly.

Reply with a JSON object only:
{"route": "vectorstore" | "off_topic", "reason": "short reason"}

- "vectorstore": question is about defence, military, NATO, doctrine, security, weapons,
   geopolitics of conflict, or asks for factual info found in such documents.
- "off_topic": greetings, small talk, or clearly unrelated topics (cooking, sports, coding)."""

REWRITE_SYSTEM = """You improve retrieval for a defence knowledge base.
Given the user question, write a single focused search query optimised for finding the answer
in military/defence documents. Expand acronyms and domain jargon. Keep it to one sentence.
Reply with a JSON object only: {"query": "..."}"""

GRADE_SYSTEM = """You judge whether each retrieved document passage is relevant to the user's
question in a defence & national security RAG system. A passage is relevant if the answer to
the question is (partly or fully) present in it.

Given the question and a numbered list of passages, reply with JSON only:
{"relevant_indices": [0, 2], "reason": "..."}
Use the 0-based indices of passages that are relevant. Empty list when none are relevant."""

GENERATE_HISTORY = """Relevant conversation history (earlier turns); answer the current question using
history only when it references prior context, otherwise focus on the retrieved passages."""

ANSWER_GUIDED = """A previous answer was deemed unhelpful. Regenerate the answer being very explicit:
state what the sources do and do not cover, and clearly flag any part that is outside the
retrieved context."""


def router_prompt(question: str) -> list:  # list[dict]
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": question},
    ]


def rewrite_prompt(question: str) -> list:
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": question},
    ]


def grade_prompt(question: str, passages: list[str]) -> list:
    body = "\n\n".join(f"[{i}] {p[:1200]}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": GRADE_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{body}"},
    ]


def generate_prompt(
    question: str,
    context: str,
    chat_history: str | None = None,
    guided: bool = False,
) -> list:
    history = f"\n{GENERATE_HISTORY}\n{chat_history}" if chat_history else ""
    guidance = ANSWER_GUIDED if guided else ""
    user = (
        f"{history}\n\nQuestion: {question}\n\nContext:\n{context}\n\n"
        f"Answer with citations to the sources above.\n{guidance}".strip()
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def hallucination_prompt(question: str, generation: str, context: str) -> list:
    return [
        {"role": "system", "content": (
            "You detect hallucinations in a RAG answer. Given the question, the generated "
            "answer, and the source context, decide whether the ANSWER is fully supported by "
            "the CONTEXT. Reply with JSON only: {\"supported\": true|false}"
        )},
        {"role": "user", "content": (
            f"Question: {question}\n\nAnswer: {generation}\n\nContext:\n{context}"
        )},
    ]


def answer_relevance_prompt(question: str, generation: str) -> list:
    return [
        {"role": "system", "content": (
            "You judge answer quality for a defence RAG system. Decide whether the answer "
            "directly addresses the user's question (regardless of correctness). "
            "Reply with JSON only: {\"useful\": true|false}"
        )},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {generation}"},
    ]


def off_topic_reply(question: str) -> str:
    return (
        "I'm a national security / defence RAG assistant — I answer questions grounded in a "
        "corpus of unclassified defence and security documents (NATO, US DoD, air force "
        "doctrine, defence policy). That question is outside my scope. "
        "Ask me something like \"What does NATO's 2022 Strategic Concept say about deterrence?\""
    )