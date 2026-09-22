from __future__ import annotations

import json
from pathlib import Path

from defence_rag.config import get_settings
from defence_rag.evaluation.ragas_compat import import_ragas
from defence_rag.evaluation.testset import load_testset
from defence_rag.graph.build import build_graph
from defence_rag.graph.state import initial_state
from defence_rag.llm import build_chat_with_fallbacks
from defence_rag.vectorstore import DefenceVectorStore


def _runtime_dir() -> Path:
    return get_settings().eval_dir / "runs"


def run_graph_on_rows(rows: list[dict], shard_size: int | None = None) -> list[dict]:
    """Run each unresolved testset row through the graph; persist per-row JSON.

    Idempotent: rows already saved under eval/runs/{id}.json are skipped, so
    interrupted runs resume without re-spending the free model budget.
    """
    settings = get_settings()
    run_dir = _runtime_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = rows[: shard_size or len(rows)]
    pending = [r for r in rows if not (run_dir / f"{r['id']}.json").exists()]
    if not pending:
        return [json.loads((run_dir / f"{r['id']}.json").read_text()) for r in rows]

    llm = build_chat_with_fallbacks()
    store = DefenceVectorStore(settings)
    graph = build_graph(llm, store, settings=settings)
    results = []
    try:
        for row in pending:
            result = graph.invoke(initial_state(row["question"]), {"recursion_limit": 20})
            saved = {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "reference_contexts": row.get("reference_contexts", []),
                "answer": result.get("generation", ""),
                "retrieved_contexts": [
                    d.page_content for d in result.get("documents", [])
                ],
                "route": result.get("route"),
                "supported": result.get("supported"),
                "useful": result.get("useful"),
                "guard_blocked": result.get("guard_blocked"),
            }
            (run_dir / f"{row['id']}.json").write_text(json.dumps(saved))
            results.append(saved)
    finally:
        store.close()
    return results


def load_runs() -> list[dict]:
    run_dir = _runtime_dir()
    if not run_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))]


def compute_metrics(runs: list[dict] | None = None) -> dict:
    """Score the saved runs with RAGAS on the configured (free) LLM."""
    from ragas.metrics.collections import answer_relevancy, context_precision, faithfulness

    runs = runs if runs is not None else load_runs()
    if not runs:
        return {}

    settings = get_settings()
    llm = build_chat_with_fallbacks()

    ragas = import_ragas()
    samples = []
    for r in runs:
        samples.append(
            ragas.dataset_schema.SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["reference_answer"],
            )
        )
    dataset = ragas.EvaluationDataset(samples=samples)
    result = ragas.evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
    )
    return result.to_pandas().to_dict("records")