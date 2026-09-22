"""Staged, budget-aware evaluation pipeline.

Usage:
    uv run scripts/evaluate.py testset [--samples 20]
    uv run scripts/evaluate.py run [--shard 6]
    uv run scripts/evaluate.py metrics
    uv run scripts/evaluate.py all [--shard 6]
"""

from __future__ import annotations

import json

import click

from defence_rag.config import get_settings
from defence_rag.evaluation.evaluate_shard import (
    compute_metrics,
    load_runs,
    run_graph_on_rows,
)
from defence_rag.evaluation.testset import build_testset, load_testset, save_testset


@click.group()
def cli(): ...


@cli.command()
@click.option("--samples", default=None, type=int)
def testset(samples: int | None):
    rows = build_testset(n_samples=samples)
    save_testset(rows)
    click.echo(f"wrote {len(rows)} testset rows -> {get_settings().eval_dir / 'testset.jsonl'}")


@cli.command()
@click.option("--shard", default=6, type=int, help="Max rows to run this invocation")
def run(shard: int):
    rows = load_testset()
    if not rows:
        click.echo("no testset; run `scripts/evaluate.py testset` first", err=True)
        raise SystemExit(1)
    results = run_graph_on_rows(rows, shard_size=shard)
    click.echo(f"ran/loaded {len(results)} runs ({len([r for r in results if r.get('answer')])} answered)")


@cli.command()
def metrics():
    results = compute_metrics()
    if not results:
        click.echo("no runs yet; run `scripts/evaluate.py run` first", err=True)
        raise SystemExit(1)
    out = get_settings().eval_dir / "ragas_scores.json"
    out.write_text(json.dumps(results, indent=2))
    averaged = {
        k: sum(float(r[k]) for r in results) / len(results)
        for k in ("faithfulness", "answer_relevancy", "context_precision")
    }
    click.echo(f"wrote {out}")
    click.echo(json.dumps({k: round(v, 4) for k, v in averaged.items()}))


@cli.command()
@click.option("--shard", default=6, type=int)
def all(shard: int):
    run([shard])
    runs = load_runs()
    click.echo(f"{len(runs)} runs persisted; computing metrics...")
    metrics()


cli.add_command(testset)
cli.add_command(run)
cli.add_command(metrics)
cli.add_command(all)


if __name__ == "__main__":
    cli()