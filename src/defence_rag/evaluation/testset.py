from __future__ import annotations

import hashlib
import json

from defence_rag.config import get_settings
from defence_rag.documents import load_sources, split_documents


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_testset(n_samples: int | None = None) -> list[dict]:
    """Extract (question, reference_answer) rows from the source documents.

    Deterministic across runs: the rows are a fixed selection of summaries of
    the corpus paragraphs, reworded into questions. Counts as seed data for a
    future LLM-generated testset (evol quality) without spending model budget
    right now.
    """
    settings = get_settings()
    docs = load_sources(settings.data_dir) if settings.data_dir.exists() else []
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    chunks = chunks[: n_samples or len(chunks)]

    rows = []
    for idx, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue
        question = f"What does {chunk.metadata.get('title', 'the document')} say about: {text[:120]}...?"
        rows.append(
            {
                "id": f"{_hash(text)}_{idx}",
                "question": question,
                "reference_answer": text,
                "reference_contexts": [text],
            }
        )
    return rows


def save_testset(rows: list[dict], path=None) -> list[dict]:
    path = path or (get_settings().eval_dir / "testset.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def load_testset(path=None) -> list[dict]:
    path = path or (get_settings().eval_dir / "testset.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]