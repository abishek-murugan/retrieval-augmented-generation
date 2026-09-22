from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    responses: list[str]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = self.responses.pop(0) if self.responses else "{}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class ZeroEmbeddings(Embeddings):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0] * self.dim


class BoundedEmbeddings(ZeroEmbeddings):
    def __init__(self, left: list[str], dim: int = 384):
        super().__init__(dim)
        self.left = left

    def _vec(self, text: str, slot: int) -> list[float]:
        vec = [0.0] * self.dim
        vec[slot % self.dim] = 1.0
        return vec


def scripted(*responses: str) -> ScriptedChatModel:
    return ScriptedChatModel(responses=list(responses))