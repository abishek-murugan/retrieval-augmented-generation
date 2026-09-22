import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from qdrant_client import QdrantClient

import pytest

from defence_rag.config import Settings
from defence_rag.guards import DefenceGuards
from defence_rag.vectorstore import DefenceVectorStore
from tests.fakes import BoundedEmbeddings, ZeroEmbeddings


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.qdrant_collection = "defence_kb_test"
    s.top_k = 2
    s.max_iterations = 2
    return s


@pytest.fixture
def guards() -> DefenceGuards:
    return DefenceGuards()


@pytest.fixture
def store(settings: Settings):
    s = DefenceVectorStore(settings, QdrantClient(":memory:"))
    yield s
    s.close()


@pytest.fixture
def seeded_store(settings: Settings):
    from langchain_core.documents import Document

    store = DefenceVectorStore(settings, QdrantClient(":memory:"))
    docs = [
        Document(
            page_content="NATO relies on collective defence and deterrence against Russia.",
            metadata={"source": "nato.pdf", "page": 1, "title": "NATO", "chunk_id": 0},
        ),
        Document(
            page_content="The US Air Force executes air superiority and long-range strike missions.",
            metadata={"source": "afdp1.pdf", "page": 2, "title": "AFDP-1", "chunk_id": 1},
        ),
    ]
    store.create_collection(384)
    store.add_documents(docs, [[1.0] * 384, [1.0] * 384])
    yield store
    store.close()


@pytest.fixture
def embeddings() -> ZeroEmbeddings:
    return ZeroEmbeddings()


@pytest.fixture
def field_embeddings() -> BoundedEmbeddings:
    return BoundedEmbeddings(["nato"])