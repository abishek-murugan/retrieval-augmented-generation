import logging

from langchain_core.language_models import BaseChatModel

from defence_rag.config import Settings
from defence_rag.guards import DefenceGuards
from defence_rag.parsing import extract_json, get_bool, get_int_list, get_str
from defence_rag.prompts import (
    answer_relevance_prompt,
    generate_prompt,
    grade_prompt,
    hallucination_prompt,
    off_topic_reply,
    rewrite_prompt,
    router_prompt,
)
from defence_rag.vectorstore import DefenceVectorStore

logger = logging.getLogger(__name__)


def ask_json(llm: BaseChatModel, messages: list[dict], default: dict | None = None) -> dict | None:
    try:
        response = llm.invoke(messages)
        content = getattr(response, "content", "") or ""
        return extract_json(str(content))
    except Exception as exc:  # noqa: BLE001 - model endpoints are flaky on free tier
        logger.warning("LLM JSON call failed: %s", exc)
        return default


class RagNodes:
    def __init__(self, llm: BaseChatModel, store: DefenceVectorStore,
                 guards: DefenceGuards, settings: Settings, embeddings=None):
        self.llm = llm
        self.store = store
        self.guards = guards
        self.settings = settings
        self.embeddings = embeddings

    # ------------------------------------------------------------- guard entry

    def input_guard(self, state) -> dict:
        passed, reasons = self.guards.check_input(state["query"])
        if not passed:
            message = (
                "Blocked by input guard. Reason: "
                + "; ".join(reasons)
                + ". This assistant answers national security / defence questions only "
                "and does not process PII, prompt injections, or off-scope requests."
            )
            return {"guard_blocked": True, "guard_reasons": reasons, "generation": message}
        return {"guard_blocked": False, "guard_reasons": []}

    # -------------------------------------------------------------- routing

    def router(self, state) -> dict:
        data = ask_json(self.llm, router_prompt(state["query"])) or {}
        route = get_str(data, ["route", "class"], "vectorstore").strip().lower()
        if route not in {"vectorstore", "off_topic"}:
            route = "vectorstore"
        return {"route": route}

    def off_topic(self, state) -> dict:
        return {"generation": off_topic_reply(state["query"])}

    # -------------------------------------------------------------- retrieval

    def rewrite(self, state) -> dict:
        data = ask_json(self.llm, rewrite_prompt(state["query"])) or {}
        new_query = get_str(data, ["query", "rewritten_query"], state["query"]) if data else state["query"]
        return {
            "rewritten_query": new_query or state["query"],
            "loop_count": state.get("loop_count", 0) + 1,
        }

    def retrieve(self, state) -> dict:
        query = state.get("rewritten_query") or state["query"]
        embeddings = self.embeddings
        if embeddings is None:
            from defence_rag.llm import get_cached_embeddings

            embeddings = get_cached_embeddings()
        dense = embeddings.embed_query(query)
        docs = self.store.hybrid_search(query, dense, k=self.settings.top_k)
        context, citations = self._format_context(docs)
        return {"documents": docs, "context": context, "citations": citations}

    @staticmethod
    def _format_context(docs) -> tuple[str, list[dict]]:
        blocks = []
        citations = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            blocks.append(f"[{i}] {doc.page_content}")
            citations.append({
                "idx": i,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", ""),
                "title": meta.get("title", ""),
            })
        context = "\n\n".join(blocks)
        return context, citations

    def grade_documents(self, state) -> dict:
        docs = state.get("documents", [])
        passages = [d.page_content[:1200] for d in docs]
        data = ask_json(self.llm, grade_prompt(state["query"], passages))
        if data is None:
            return {"no_relevant": not bool(docs)}
        idxs = [i for i in get_int_list(data, ["relevant_indices", "indices"]) if i < len(docs)]
        relevant = [docs[i] for i in idxs]
        context, citations = self._format_context(relevant)
        return {
            "documents": relevant,
            "context": context,
            "citations": citations,
            "no_relevant": not bool(relevant),
        }

    # ---------------------------------------------------------------- generate

    def generate(self, state) -> dict:
        history = self._history_block(state)
        messages = generate_prompt(
            state["query"],
            state.get("context", ""),
            chat_history=history,
            guided=state.get("guided", False),
        )
        answer = self._ask_text(messages)
        return {"generation": answer}

    def _ask_text(self, messages: list[dict], max_tokens: int | None = None) -> str:
        response = self.llm.invoke(messages, {"max_tokens": max_tokens} if max_tokens else {})
        content = getattr(response, "content", "") or ""
        return str(content).strip()

    def _history_block(self, state) -> str | None:
        chat_history = state.get("chat_history") or []
        if not chat_history:
            return None
        lines = [f"User: {turn.get('query')}\nAssistant: {turn.get('generation')}"
                 for turn in chat_history[-6:]]
        return "\n\n".join(lines)

    # ------------------------------------------------------------- validation

    def hallucination_check(self, state) -> dict:
        data = ask_json(
            self.llm,
            hallucination_prompt(state["query"], state.get("generation", ""), state.get("context", "")),
        ) or {}
        supported = get_bool(data, ["supported", "grounded", "faithful"])
        return {"supported": supported}

    def answer_check(self, state) -> dict:
        data = ask_json(
            self.llm,
            answer_relevance_prompt(state["query"], state.get("generation", "")),
        ) or {}
        useful = get_bool(data, ["useful", "relevant", "acceptable"])
        return {"useful": useful}

    def finalize(self, state) -> dict:
        passed, reasons = self.guards.check_output(state.get("generation", ""))
        chat_history = list(state.get("chat_history") or [])
        chat_history.append({"query": state["query"], "generation": state.get("generation", "")})
        history = chat_history[-10:]
        return {
            "output_guard_fail": not passed,
            "guard_reasons": reasons,
            "chat_history": history,
        }

    def no_answer(self, state) -> dict:
        docs = state.get("documents", [])
        hint = ""
        if docs:
            hint = " The retrieved passages did not address this question."
        return {
            "generation": (
                "I don't know. The knowledge base does not contain enough information to answer "
                f"this question.{hint}"
            )
        }