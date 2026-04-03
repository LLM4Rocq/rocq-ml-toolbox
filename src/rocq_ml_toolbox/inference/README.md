# Inference

FastAPI + uvicorn service that manages multiple `pet-server` workers via Redis. The stack is:
- `rocq-ml-server` CLI (process supervisor)
- `rocq_ml_toolbox.inference.arbiter` (worker lifecycle/restarts)
- `rocq_ml_toolbox.inference.server` (HTTP API)

## Requirements
- `redis-server` executable on `PATH` (spawned by `rocq-ml-server`).
- `pet-server` from `pytanque` on `PATH` (or pass `--pet-server-cmd`).
- Python deps via `pip install -e .[server]`.
- Rocq/Coq toolchain (`coqc`, `coq-lsp`) for dump extraction and compilation checks.

## Run the server
```bash
rocq-ml-server --num-pet-server 4 --workers 9 --port 5000
```

Detached mode:

```bash
rocq-ml-server -d --pidfile rocq-ml-server.pid --arbiter-log arbiter.log
```

In detached mode, extra pidfiles are written next to the base pidfile (`.uvicorn`, `.arbiter`, `.redis`).

## Python client
```python
from rocq_ml_toolbox.inference.client import PytanqueExtended

client = PytanqueExtended("127.0.0.1", 5000)
client.connect()  # creates/loads a session via /login

state = client.get_state_at_pos("/path/to/file.v", line=10, character=0)
state = client.run(state, "intros.")
print(client.goals(state))

report = client.safeverify(
    source="/path/to/Source.v",
    target="/path/to/Target.v",
    root="/path/to/project",
)
print(report["summary"])
```

## API Surface
- `GET /health`: aggregated health snapshot (arbiter heartbeat + worker states).
- `GET /login`: create a new session id.
- `POST /rpc`: main Petanque route gateway (`route_name`, `params`, `timeout`).
- `POST /get_dump`: stream AST/proof/diagnostic dump for a `.v` file.
- `POST /get_glob`: load/compile and return `.glob` data.
- `POST /safeverify`: run SafeVerify in the server environment.
- `POST /tmp_file`: allocate a temporary `.v` file path.
- `POST /access_libraries`: load `<coq_lib>/<env>.toc.json` or fallback-scan `theories` + `user-contrib`.
- `POST /read_file`: chunked UTF-8 file reads (`offset`, `max_chars`).
- `POST /write_file`: chunked writes (`offset`, `truncate`) with startup fs policy enforcement.
- `POST /read_docstrings`: load docstring entries from `<source>.toc.json`.

Most proof interaction calls (`run`, `goals`, `start`, `state_hash`, etc.) go through `/rpc` and the `pytanque` route registry.

## Useful CLI Flags
- `--num-pet-server`, `--pet-server-start-port`: worker pool sizing and first port.
- `--soft-max-ram-per-pet`, `--hard-max-ram-per-pet`: restart thresholds in MB.
- `--redis-port`: local Redis port used by this server.
- `--session-ttl-seconds`: inactivity TTL before session eviction.
- `--session-cleanup-interval-seconds`: eviction scan interval.
- `--session-cache-keep-feedback`: keep `State.feedback` in cache (off by default).
- `--pet-server-cmd`: override `pet-server` executable path.
- `--fs-access-mode`: immutable file policy for read/write endpoints (`read_lib_only` or `rw_anywhere`).
- `--coq-lib-path`: optional override for Coq lib root (otherwise resolved from `coqc -where`).
