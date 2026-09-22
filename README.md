# Defence RAG

A production-grade, agentic Retrieval-Augmented Generation service for publicly
available defence doctrine, built to run **entirely on free tiers** (zero spend).

Pipeline: user → Guardrails input validation → LangGraph self-corrective agent
(rewrite → hybrid RAG → retrieve → grade → generate → groundedness → usefulness)
→ Guardrails output validation → traced answer with citations.

## Architecture

```
User / API
   │  GET /health, POST /ask
   ▼
FastAPI ──────────────────────────────────────────────┐
   │                                                  │
   ▼                                                  ▼
Guardrails AI                                Qdrant (dense vectors)
   │  input guard (injection/PII/off-topic)   +   BM25 (lexical)
   ▼                                                  │
LangGraph agent  ──►  dense query → RRF fusion ←──────┘
  router → rewrite → retrieve → grade-docs   （relevance loop ≤ N）
         → generate → hallucination-check → usefulness-check
   │
   ├─ ChatOpenAI ─► Portkey Cloud ─► OpenRouter free models (primary + fallbacks)
   └─ LangSmith tracing (traces/evals)   ◀────────── Portkey observability
```

- **LLM transport**: `ChatOpenAI` wrapper → Portkey Cloud (`@openrouter`
  integration) → OpenRouter free tier (50 req/day). Primary
  `nvidia/nemotron-3-super-120b-a12b:free` with a 3-model fallback chain
  (`RunnableWithFallbacks`).
- **Retrieval**: Qdrant hybrid — dense (BGE-small via fastembed, disk-cached)
  + lexical BM25 (rank-bm25) fused with Reciprocal Rank Fusion.
- **Self-correction**: relevance grader may rewrite the query and re-retrieve
  until relevant docs are found or a loop budget is exhausted; a hallucination
  guard checks the answer against the retrieved context; a usefulness check
  may trigger one final rewrite.
- **Guardrails**: custom validators (prompt-injection, PII, off-topic,
  citation-requirement, toxicity). Safe answers exit the graph.
- **Evaluation**: RAGAS 0.4 (faithfulness, answer_relevancy, context_precision)
  over a staged/resumable sharded pipeline that respects the free daily budget.
- **Observability**: LangSmith for traces + RAGAS eval; Portkey for LLM
  observability. Both free-account features.

## Layout

```
src/defence_rag/
  config.py       env/settings (pydantic-settings) + observability env
  llm.py          Portkey→OpenRouter factory, fallbacks, cached embeddings
  guards.py       Guardrails AI custom validators
  documents.py    PDF loading + chunking
  vectorstore.py  Qdrant dense + BM25 + RRF hybrid store
  prompts.py      router/rewrite/grade/generate/groundedness prompts
  parsing.py      strict JSON extraction helpers
  graph/          LangGraph state, nodes, conditional edges, build
  api/main.py     FastAPI app (/health, /ask)
  evaluation/     ragas_compat shim, testset, sharded evaluate
scripts/
  fetch_sources.py   download public docs (with wayback fallback)
  ingest.py          build/refresh the Qdrant collection
  evaluate.py        testset | run | metrics | all  (resumable)
  serve.py           run the FastAPI service
manifests/defence_sources.json
tests/              guards, graph (scripted LLM), parsing, retrieval
compose.yaml        qdrant + api stack        (.env required, see .env.example)
Dockerfile
```

## Quickstart

```bash
cp .env.example .env        # fill OPENROUTER_API_KEY, PORTKEY_API_KEY, LANGSMITH_API_KEY
docker compose up -d qdrant # or any Qdrant on QDRANT_URL
uv sync
uv run python scripts/fetch_sources.py
uv run python scripts/ingest.py           # ~170 chunks (cached embeddings)
uv run pytest
uv run python scripts/serve.py            # http://localhost:8000/docs
```

```bash
uv run python scripts/evaluate.py testset --samples 20   # free (local extraction)
uv run python scripts/evaluate.py run --shard 6          # spends few free tokens
uv run python scripts/evaluate.py metrics                # RAGAS scores
```

## Config (env)

| Variable | Default |
| --- | --- |
| `OPENROUTER_API_KEY` | – |
| `PORTKEY_API_KEY`, `PORTKEY_PROVIDER` | `@openrouter` |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | `defence-rag` |
| `PRIMARY_MODEL` / `FALLBACK_MODELS` | free Nemotron / 3 fallbacks |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `QDRANT_URL`, `QDRANT_COLLECTION` | `http://localhost:6333` / `defence_kb` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | 800 / 120 / 4 |
| `MAX_ITERATIONS` | 3 |

All source documents are **public/unclassified** (NATO Strategic Concept 2022,
DoD Instruction 3000.09, AFDP-1 Air Force Doctrine Publication).