from __future__ import annotations

import uuid

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from defence_rag.config import Settings


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


class DefenceVectorStore:
    """Qdrant dense store + in-process BM25 lexical index, fused with RRF."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.settings = settings
        self._owns_client = client is None
        self.client = client or QdrantClient(
            url=settings.qdrant_url, check_compatibility=False
        )
        self.collection = settings.qdrant_collection
        self._bm25 = None
        self._bm25_docs: list[Document] = []

    @property
    def exists(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:
            return False

    def create_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE)
                },
            )

    def delete_collection(self) -> None:
        if self.exists:
            self.client.delete_collection(self.collection)

    def add_documents(self, docs: list[Document], dense_vectors: list[list[float]]) -> int:
        points = []
        for doc, dense in zip(docs, dense_vectors, strict=True):
            doc_id = uuid.uuid5(uuid.NAMESPACE_URL, doc.page_content[:200] + str(doc.metadata))
            points.append(
                qm.PointStruct(
                    id=str(doc_id),
                    vector={"dense": dense},
                    payload={"text": doc.page_content, "metadata": doc.metadata},
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)
        self._bm25 = None
        return len(points)

    @property
    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count

    # ------------------------------------------------------- hybrid retrieval

    def _bm25_index(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            records = self.client.scroll(
                collection_name=self.collection, limit=10_000, with_payload=True
            )[0]
            self._bm25_docs = [self._point_to_document(p) for p in records]
            self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in self._bm25_docs])
        return self._bm25

    def hybrid_search(self, query: str, dense_vector: list[float],
                      k: int | None = None) -> list[Document]:
        k = k or self.settings.top_k
        dense = self.client.query_points(
            collection_name=self.collection,
            query=dense_vector,
            using="dense",
            limit=k * 3,
            with_payload=True,
        ).points

        dense_hits = {self._key(p): (self._point_to_document(p), i + 1)
                      for i, p in enumerate(dense)}
        bm25 = self._bm25_index()
        bm25_scores = bm25.get_scores(_tokenize(query))
        bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        rrf: dict[str, float] = {}
        for rank, (key, _) in enumerate(dense_hits.items()):
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (60 + rank)
        lexical_rank = {i: r for r, i in enumerate(bm25_order)}
        for i in bm25_order[: k * 2]:
            key = self._doc_key(self._bm25_docs[i])
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (60 + lexical_rank[i])

        merged = []
        for key, score in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True):
            if key in dense_hits:
                merged.append(dense_hits[key][0])
                continue
            idx = bm25_order[0]
            for i, doc in enumerate(self._bm25_docs):
                if self._doc_key(doc) == key:
                    merged.append(doc)
                    break
        return merged[:k]

    @staticmethod
    def _key(point) -> str:
        return str(point.id) or ""

    @staticmethod
    def _doc_key(doc: Document) -> str:
        return str(doc.metadata.get("chunk_id", doc.metadata.get("page", ""))) + doc.metadata.get("source", "")

    @staticmethod
    def _point_to_document(point) -> Document:
        payload = point.payload or {}
        return Document(
            page_content=payload.get("text", ""),
            metadata=payload.get("metadata", {"source": "unknown", "page": 0}),
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()