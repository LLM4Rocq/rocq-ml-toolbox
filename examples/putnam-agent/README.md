# Putnam Pydantic-AI Agents

This benchmark folder ships with one agent implementation in
`agent/pydantic_agent.py`:

- `build_scalable_putnam_agent`: designed for horizontal scale (many independent agent sessions with `ScalablePutnamRunner`) and with `safe_verify` required at `end`.

It uses one `PytanqueExtended` client per agent session and initializes the proof state with
`get_state_at_pos` at the end of `Proof.`.

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
  --port 5000
```

Notes:

- Default problem: `examples/putnam-agent/putnam/mathcomp/putnam_1965_a5.v`
- Default model: `moonshotai/kimi-k2.5`
- Logs are enabled by default for real-time feedback (`--quiet` to disable)

## DocQ Agent Quickstart

This second agent workflow focuses on library/docstring exploration plus intermediate-lemma insertion on a virtual document DAG.

```bash
OPENROUTER_API_KEY=sk-or-... \
DOCQ_SEARCH_BASE_URL=http://127.0.0.1:9000 \
python examples/putnam-agent/run_docq_agent_openrouter.py \
  --source /home/rocq/.opam/4.14.2+flambda/lib/coq/user-contrib/MathComp/ssreflect/ssrbool.v \
  --env coq-mathcomp \
  --host 127.0.0.1 \
  --port 5000
```

Optional semantic search env vars:

- `DOCQ_SEARCH_BASE_URL` (required to enable `semantic_doc_search`)
- `DOCQ_SEARCH_ROUTE` (default: `/search`)
- `DOCQ_SEARCH_API_KEY` (optional bearer token)
