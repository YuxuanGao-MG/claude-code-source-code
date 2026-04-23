# openmesh-bench — Handoff to Local Claude Code

Context for a fresh Claude Code session picking up this work. Read this in full before touching code.

## What this project is

A thin OpenAI-compatible client + tracing layer used to run agent benchmarks
(GAIA, tau-bench, SWE-bench Pro) across 8 frontier/mid-tier LLMs. All upstream
calls route through **OpenRouter** so spend lands on **OpenMesh Inc.**'s books
as cost of goods sold; the user self-pays OpenMesh for the benchmark compute,
which produces revenue on OpenMesh's income statement.

The package lives in `openmesh-bench/` on branch
`claude/openmesh-llm-wrapper-VN3EA`. The surrounding repo is unrelated
(it's the Claude Code source mirror) — don't touch anything outside
`openmesh-bench/`.

## Critical: secrets hygiene

During this conversation the user pasted live OpenRouter and HF tokens into
chat. **Both have been instructed to be rotated.** Before doing anything that
needs credentials:

1. Confirm the user has revoked the old keys.
2. New keys go in `openmesh-bench/.env` (gitignored) — never commit them.
3. For unattended runs, use GitHub Actions/Codespaces secrets, not files.

Old (rotated, do not use): `sk-or-v1-0bdf...f0f65954`, `hf_...SPTo`.

## Decisions already locked in

| Decision | Choice | Rationale |
|---|---|---|
| Upstream gateway | **OpenRouter** | Single account vs. 6 direct providers. ~5% markup acceptable for the time saved. Migration to direct accounts (Anthropic/OpenAI/Google) deferred to "after research burn." |
| Provider router lib | None — straight `openai` SDK with OpenRouter base URL | No need for LiteLLM since OpenRouter normalizes. |
| Trace storage | **Hugging Face Datasets**, public dataset, batch upload | Free, research-community-native, has a viewer. Cloudflare R2 was considered and rejected (more setup, not needed yet). |
| Streaming | **Off by default**, togglable via env / CLI / per-call | Non-streaming gives `usage` reliably; streaming adds TTFT but risks gateway dropping `usage`. |
| Models | 8 (4 frontier + 4 mid). Kimi is **K2.6** (now available on OpenRouter). | See `registry.py`. |
| Budget | **$5–20k USD total** across all benchmarks | OpenRouter funded with $200 to start, auto-refill +$200 at <$50. |
| Benchmarks | GAIA, tau-bench, SWE-bench Pro — in that order of difficulty | None wired yet. |
| Public API | **Deferred** — no FastAPI / no public `/v1/runs` endpoint for now. CLI-only. | Don't build the productized API until the research burn is done. |

## Current code state (committed on the branch)

```
openmesh-bench/
├── pyproject.toml              # deps: openai, httpx, python-dotenv, huggingface_hub
├── .env.example                # template; copy to .env locally
├── .gitignore                  # excludes .env, traces/, __pycache__, etc.
└── openmesh_bench/
    ├── __init__.py             # public API exports
    ├── config.py               # env loading; defaults base_url to OpenRouter
    ├── registry.py             # 8 ModelSpec entries with placeholder OpenRouter ids + prices
    ├── accounting.py           # Tracer (JSONL), trace_session ContextVar, BudgetExceeded, resume support
    ├── client.py               # TracingClient — openai SDK shape (.chat.completions.create)
    ├── smoke.py                # `openmesh-smoke` CLI: checks /models then 1 call per model
    └── uploader.py             # `openmesh-upload <run_dir>` — pushes to HF dataset
```

### What works
- Non-streaming and streaming paths in `client.py` (streaming reassembles tool-call deltas).
- Trace JSONL captures full request, full response, usage, cost, latency, ttft (when streaming), upstream provider, retries.
- Spend caps: per-task / per-model / per-run, raised as `BudgetExceeded`.
- Task resume via `completed_tasks.jsonl`.
- HF uploader writes per-run `meta.json` (summary stats) and updates a top-level `index.json` listing every run.

### What's NOT done (in priority order)
1. **Smoke test against the real OpenRouter** — model IDs in `registry.py` are educated guesses. Run `openmesh-smoke` and fix any IDs that don't match OpenRouter's `/v1/models` listing.
2. **Verify pricing** in `registry.py` against OpenRouter's published prices. Spend caps depend on these. They're currently placeholders — likely roughly right for frontier, may be wrong for mid.
3. **GAIA harness** — load the dataset (HF: `gaia-benchmark/GAIA`, gated; user must accept license), iterate tasks, call `TracingClient` with `trace_session(task_id)`, score with the bundled GAIA scorer.
4. **tau-bench wiring** — clone `sierra-research/tau-bench`, swap their LLM caller for `TracingClient` (it accepts the same `chat.completions.create` shape).
5. **SWE-bench Pro wiring** — significantly more work (Docker, repo clones, test execution). Defer until 1–4 are solid.
6. **GitHub Actions workflow** for unattended runs (nice-to-have).

## How to verify the code locally

```bash
cd openmesh-bench
python -m py_compile openmesh_bench/*.py    # syntax check
pip install -e .
cp .env.example .env                          # then edit .env with real keys
openmesh-smoke --models gemini-3-flash        # cheapest model first
```

Expected output: list of 8 models from OpenRouter (or warning), one OK line per
tested model, total spend printed, traces in `traces/smoke-<ts>/calls.jsonl`.

Then:
```bash
openmesh-upload traces/smoke-<ts>
```
Prints a `huggingface.co/datasets/<repo>/...` URL.

## Runtime options for the actual burn

User has three choices, no decision made yet:

- **A. Local laptop** — fine for smoke + small GAIA slice.
- **B. GitHub Codespace** — secrets via repo settings → Codespaces. Browser-based.
- **C. GitHub Actions** — secrets via repo settings → Actions, `workflow_dispatch` trigger. Best for multi-hour burns.

If the user picks C, scaffold `.github/workflows/bench.yml` with inputs:
`benchmark` (gaia|tau-bench|swe-bench-pro), `models` (comma-separated or "all"),
`max_usd_per_run`, `task_filter`. Workflow runs `openmesh-bench`, then
`openmesh-upload`, then posts a summary as a workflow annotation.

## Open questions for the user (ask before assuming)

1. **HF namespace.** Is there an `openmesh` org on Hugging Face under OpenMesh
   Inc., or should the dataset go under the user's personal namespace? Default
   in `.env.example` is `openmesh/bench-traces` — change if needed.
2. **OpenRouter payment source.** Was OpenRouter funded with the OpenMesh corp
   card or personal card? If personal, the revenue-attribution story doesn't
   work yet — flag this to the user, don't silently proceed.
3. **`OPENMESH_APP_URL` / `APP_NAME`.** Defaulted to `https://openmesh.ai` /
   `openmesh-bench`. These show up on OpenRouter's public leaderboard.
4. **Trace privacy.** Currently the public HF dataset will contain full prompts
   and model outputs. Fine for benchmark tasks (which are public anyway). If
   real user prompts ever flow through this, add a `private/` path or per-run
   visibility flag.
5. **GAIA license.** User must accept the dataset license on HF first or
   downloads will fail. Confirm before wiring GAIA.

## Things to not do

- Do not commit `.env` or any file containing real keys.
- Do not refactor `accounting.py` or `client.py` unnecessarily — the core trace
  schema is now load-bearing for the HF dataset format.
- Do not add provider-native features (Anthropic prompt caching, OpenAI batch
  API, Gemini grounding). The user explicitly said benchmark-only, no native
  features needed. Stay OpenAI-shape.
- Do not add LiteLLM, Portkey, or another routing layer. OpenRouter is the
  router.
- Do not stand up a FastAPI / public `/v1/runs` endpoint yet. That's Phase 2,
  after the research burn produces results.
- Do not migrate to direct provider accounts in code yet — wait until the user
  has actually opened those accounts.

## Conversation summary (for context only)

- User wants to spend $5–20k benchmarking 8 LLMs on GAIA / tau-bench /
  SWE-bench Pro. Goal: research data + revenue on OpenMesh Inc.'s books.
- OpenMesh Inc. is a registered C-corp with EIN and bank accounts. Domain
  `openmesh.ai` is on GoDaddy with a Wix marketing site (no backend).
- Considered: routing through 6 direct provider accounts. Rejected for the
  initial burn due to multi-week setup; OpenRouter chosen as faster path.
  Direct accounts may be re-evaluated later for cost / provider relationships.
- Considered: hosting traces on Cloudflare R2, Supabase, B2. Chose HF Datasets
  for batch upload simplicity and research-community discoverability.
- Considered: standing up a public API endpoint at `api.openmesh.ai`. Deferred
  until research is done.
- Tax/incorporation discussion happened but is out of scope for coding.

## Branch / commits

```
claude/openmesh-llm-wrapper-VN3EA
├── 53ff628  Add openmesh-bench: OpenAI-compatible wrapper with tracing for GAIA/tau-bench/SWE-bench Pro
└── 5f3d13e  Route via OpenRouter, narrow to 8-model set, add HF Datasets uploader
```

Push the branch after changes — don't merge to main without user approval.
