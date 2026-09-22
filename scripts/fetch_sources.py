"""Fetch public-domain defence documents into data/raw.

The manifest ships with URLs archived on the Internet Archive (web.archive.org),
which keeps the pipeline reproducible even if the original publishers move files.
Direct origins are preferred, wayback snapshots are the fallback.

Usage:
  uv run python scripts/fetch_sources.py [--manifest manifests/defence_sources.json] [--url <id>=<url>]
"""
import json
import sys
from pathlib import Path

import click
import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_MANIFEST = ROOT / "manifests" / "defence_sources.json"

UA = {"User-Agent": "defence-rag-fetcher/0.1 (portfolio; contacts public domain docs)"}


def _wayback_snapshot(url: str, client: httpx.Client) -> str | None:
    try:
        r = client.get(
            "https://archive.org/wayback/available",
            params={"url": url.replace("https://", "").replace("http://", "")},
            timeout=15,
        )
        snap = r.json().get("archived_snapshots", {}).get("closest", {})
        if snap.get("available") and snap.get("url"):
            return snap["url"].replace("/web/", "/web/", 1).replace("http://", "https://")
        return None
    except Exception:  # noqa: BLE001
        return None


def _download(url: str, dest: Path, client: httpx.Client) -> bool:
    r = client.get(url, headers=UA, follow_redirects=True, timeout=120)
    r.raise_for_status()
    if not r.content[:4] == b"%PDF":
        raise ValueError(f"Not a PDF: {url}")
    dest.write_bytes(r.content)
    return True


def fetch_one(item: dict, client: httpx.Client, force: bool) -> Path | None:
    out = RAW_DIR / f"{item['id']}.pdf"
    if out.exists() and not force:
        click.echo(f"skip {item['id']} (already present)")
        return out
    attempts = [item.get("url")] + item.get("mirrors", [])
    snapshot = _wayback_snapshot(item.get("url", ""), client)
    if snapshot:
        attempts.insert(1, snapshot)
    for url in attempts:
        try:
            _download(url, out, client)
            click.echo(f"ok   {item['id']}  <- {url[:90]}")
            return out
        except Exception as exc:  # noqa: BLE001
            click.echo(f"fail {item['id']} from {url[:70]}: {type(exc).__name__}")
    return None


@click.command()
@click.option("--manifest", type=click.Path(dir_okay=False), default=str(DEFAULT_MANIFEST))
@click.option("--url", multiple=True, help="Extra source as id=https://...")
@click.option("--force", is_flag=True)
def main(manifest: str, url: tuple[str, ...], force: bool):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    items = json.loads(manifest_path.read_text())
    extra = []
    for spec in url:
        if "=" in spec:
            key, _, value = spec.partition("=")
            extra.append({"id": key, "url": value})
    items.extend(extra)

    with httpx.Client(timeout=30) as client:
        ok = [fetch_one(item, client, force) for item in items]

    fetched = sum(1 for p in ok if p)
    click.echo(f"\nFetched {fetched}/{len(items)} documents into {RAW_DIR}")
    sys.exit(0 if fetched == len(items) else 1)


if __name__ == "__main__":
    main()