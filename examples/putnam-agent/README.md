# Putnam Pydantic-AI Agents

This benchmark folder ships with one agent implementation in
`agent/pydantic_agent.py`:

- `build_scalable_putnam_agent`: designed for horizontal scale (many independent agent sessions with `ScalablePutnamRunner`) and with `safe_verify` required at `end`.

It uses one `PytanqueExtended` client per agent session and initializes the proof state with
`get_state_at_pos` at the end of `Proof.`.
When the server runs in Docker, the session first uploads the local benchmark file via `/tmp_file`
and then works on that server-side staged path (including `safe_verify`).

## Docker Quickstart

### 1) Start the server in Docker (terminal A)

```bash
docker run --rm --network host -it \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  theostos/coq-mathcomp:9.0-2.5.0 \
  rocq-ml-server \
    --port 5000 \
    --hard-max-ram-per-pet 10000 \
    --soft-max-ram-per-pet 8000 \
    --num-pet-server 3 \
    --workers 6
```

### 2) Install client-side deps (terminal B)

```bash
pip install -e .[client]
pip install pydantic-ai
pip install git+https://github.com/llm4rocq/pytanque.git
```

### 3) Run a smoke example (terminal B)

This runs without external LLM API keys by using `pydantic-ai`'s `TestModel`.

```bash
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path("examples/putnam-agent").resolve()))

from agent.pydantic_agent import (
    PutnamAgentSession,
    PutnamBenchProblem,
    build_scalable_putnam_agent,
)
from pydantic_ai.models.test import TestModel
from rocq_ml_toolbox.inference.client import PytanqueExtended

problem = PutnamBenchProblem.from_file(
    "examples/putnam-agent/putnam/mathcomp/putnam_1980_b1.v",
    bench_root="examples/putnam-agent/putnam",
)
client = PytanqueExtended("127.0.0.1", 5000)
session = PutnamAgentSession.from_problem(client, problem)

agent = build_scalable_putnam_agent(model=TestModel(call_tools=["list_states", "get_goals"]))
print(agent.run_sync("Inspect initial proof state.", deps=session).output)
PY
```

### 4) (Optional) Run the example test suite

```bash
pytest -q examples/putnam-agent/test_pydantic_agent.py
```

## Real-time Feedback

Two minimal options:

1) Built-in console logs:

```python
session = PutnamAgentSession.from_problem(client, problem, log_enabled=True)
```

2) Custom logger function:

```python
from agent import make_console_logger

session = PutnamAgentSession.from_problem(
    client,
    problem,
    logger=make_console_logger("putnam-1980-b1"),
)
```

## Quick Sketch (Real Model)

```python
from pathlib import Path
import sys

# from repository root
sys.path.insert(0, str(Path("examples/putnam-agent").resolve()))

from agent import (
    PutnamAgentSession,
    PutnamBenchProblem,
    build_scalable_putnam_agent,
    make_console_logger,
)
from rocq_ml_toolbox.inference.client import PytanqueExtended

problem = PutnamBenchProblem.from_file(
    "examples/putnam-agent/putnam/coquelicot/putnam_1990_a1.v",
    bench_root="examples/putnam-agent/putnam",
)

client = PytanqueExtended("127.0.0.1", 5000)
session = PutnamAgentSession.from_problem(
    client,
    problem,
    logger=make_console_logger("putnam-1990-a1"),  # real-time feedback
)

agent = build_scalable_putnam_agent(model="openai:gpt-4.1-mini")
result = agent.run_sync("Prove the theorem.", deps=session)
print(result.output)
```

## Batch K Agents (Kimi K2.5 via OpenRouter)

Use the ready-made script:

```bash
OPENROUTER_API_KEY=sk-or-... \
python examples/putnam-agent/run_batch_kimi_openrouter.py \
  -k 6 \
  --host 127.0.0.1 \
  --port 5000 \
  --temperature 1.0 \
  --top-p 0.95
```

Notes:

- Default problem: `examples/putnam-agent/putnam/mathcomp/putnam_1965_a5.v`
- Default model: `moonshotai/kimi-k2.5`
- Logs are enabled by default for real-time feedback (`--quiet` to disable)

## DocQ Agent Quickstart

This second agent workflow focuses on library/docstring exploration plus intermediate-lemma insertion on a virtual document DAG.
It supports both forward and inverse edits on the virtual workspace:

- `add_import` / `remove_import`
- `add_intermediate_lemma` / `remove_intermediate_lemma`

It also exposes explicit branch/DAG controls:

- `list_docs`, `checkout_doc`, `show_doc`
- `completion_status` (quick check that current head proof is actually closed)
- all state and mutation tools accept optional `doc_id`
- phased intermediate lemma flow:
  - `prepare_intermediate_lemma`
  - `prove_intermediate_lemma`
  - `drop_pending_intermediate_lemma`
  - `list_pending_intermediate_lemmas`
  - optional `subagent_message` handoff on prepare/prove/add to pass import hints, local-goal focus, and proof strategy to the lemma sub-agent

Validation and dependency propagation notes:

- intermediate-lemma sub-agent can declare imports with `require_import(libname, source)`;
  the main agent applies these imports before registering the proved lemma.
- each sub-agent proof run uses its own fresh server client (when available), so its RPC id
  stream is isolated from the main-agent client.
- every document mutation (`add/remove import`, `add/remove lemma`) is revalidated in a fresh
  server session; rejected mutations are rolled back automatically.
- completion is enforced at output validation: pending intermediate lemmas are forbidden, and the
  head proof must have `latest_goals_count == 0`; otherwise the model is retried with an actionable hint.

TOC note:

- Some env-level TOCs expose logical module paths without `.v` suffix (for example `mathcomp/boot/ssrbool`).
- `read_source_file` now resolves both logical paths and `.v` paths automatically.
- `--env` is optional if exactly one `<coq_lib>/*.toc.json` is present in the server image.
- `--source` defaults to `examples/putnam-agent/putnam/mathcomp/putnam_1965_a5.v`.
- DocQ sessions stage virtual files with server-managed `/tmp` paths (no host-path root override).
- Real-time logs are enabled by default; use `--quiet` to disable.
- Batch mode supports `-k/--num-agents` plus `--max-concurrency`.
- By default, each run exports artifacts under
  `examples/putnam-agent/interactive_test/docq_batch_YYYYmmdd_HHMMSS/`:
  - `task.log` (runtime trace),
  - `all_messages.jsonl` / `new_messages.jsonl` (incremental pydantic-ai message history, one JSON object per line),
  - `events.jsonl` (incremental model/tool event stream, one JSON object per line),
  - `docs/doc_<id>.v`, `final_doc.v`, and `final_doc_materialized.v` (virtual document DAG snapshots + materialized head proof),
  - `summary.json` (usage, pending lemmas, head state/goals).
- Disable artifact export with `--no-artifacts`.
- Note: the DocQ workflow currently does not auto-run a global Putnam `safe_verify/end` step; check
  `summary.json` (`head_proof_finished`, `head_goals_count`) to see whether the current head workspace is closed.

```bash
OPENROUTER_API_KEY=sk-or-... \
DOCQ_SEARCH_BASE_URL=http://127.0.0.1:9000 \
python examples/putnam-agent/run_docq_agent_openrouter.py \
  -k 4 \
  --host 127.0.0.1 \
  --port 5000 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-requests 200 \
  --threshold-compression 100000 \
  --artifacts-dir examples/putnam-agent/interactive_test/docq_demo_run
```

Context compaction:

- `--threshold-compression` (default `100000`) enables automatic context compression.
- When cumulative token usage crosses the threshold during a task, the runner asks the model for a high-signal task handoff summary, then resumes with:
  `main task prompt + summary` (history reset).
- The same compression strategy is applied symmetrically to intermediate-lemma sub-agents.
- Set `--threshold-compression 0` to disable this behavior.

Optional semantic search env vars:

- `DOCQ_SEARCH_BASE_URL` (required to enable `semantic_doc_search`)
- `DOCQ_SEARCH_ROUTE` (default: `/search`)
- `DOCQ_SEARCH_API_KEY` (optional bearer token)
- `--max-requests` controls pydantic-ai `request_limit` (default: `200`).

If retrieval is not available yet and you want to hide that tool entirely:

```bash
OPENROUTER_API_KEY=sk-or-... \
python examples/putnam-agent/run_docq_agent_openrouter.py \
  --host 127.0.0.1 \
  --port 5000 \
  --disable-semantic-tool
```
