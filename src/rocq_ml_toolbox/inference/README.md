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
- For dump/dependency routes (notably `POST /get_dump`), install `fcc` + a patched `rocq-lsp` with the `proofdepsdump` plugin:
  - fork: https://github.com/theostos/rocq-lsp
  - branch by version:
    - `proofdepsdump-dump-load-v9.1`
    - `proofdepsdump-dump-load-v9.0`
    - `proofdepsdump-dump-load-v8.20`

## Run the server
```bash
rocq-ml-server --num-pet-server 4 --workers 9 --port 5000
```

Detached mode:

```bash
rocq-ml-server -d --pidfile rocq-ml-server.pid --arbiter-log arbiter.log
```

In detached mode, extra pidfiles are written next to the base pidfile (`.uvicorn`, `.arbiter`, `.redis`).

## Run with Docker

Rocq (9.0) image example:

```bash
docker run --rm -p 5000:5000 --entrypoint bash theostos/rocq-server:9.0 -lc '
set -e
export PATH=/home/rocq/miniconda/envs/rocq-ml/bin:$PATH
rocq-ml-server --host 0.0.0.0 --port 5000 --num-pet-server 2 --workers 1
'
```

Coq (8.20) image example:

```bash
docker run --rm -p 5000:5000 --entrypoint bash theostos/coq-server:8.20 -lc '
set -e
export PATH=/home/coq/miniconda/envs/rocq-ml/bin:$PATH
rocq-ml-server --host 0.0.0.0 --port 5000 --num-pet-server 2 --workers 1
'
```

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
- `POST /access_libraries`: load `<coq_lib>/<env>.toc.json` (env optional when a single TOC exists) or fallback-scan `theories` + `user-contrib`.
- `POST /read_file`: chunked UTF-8 file reads (`offset`, `max_chars`).
- `POST /write_file`: chunked writes (`offset`, `truncate`) with startup fs policy enforcement.
- `POST /read_docstrings`: load docstring entries from `<source>.toc.json`.

Most proof interaction calls (`run`, `goals`, `start`, `state_hash`, etc.) go through `/rpc` and the `pytanque` route registry.

## CLI Reference

`rocq-ml-server` (from `rocq_ml_toolbox.inference.cli`) accepts:

- Network / process:
  - `-H`, `--host` (default: `0.0.0.0`)
  - `-p`, `--port` (default: `5000`)
  - `-d`, `--detached` (default: `false`)
  - `--pidfile` (default: `rocq-ml-server.pid`, used in detached mode)
  - `--arbiter-log` (default: `arbiter.log`)
  - `-l`, `--log` (default: `false`, enable uvicorn log config)
- Uvicorn worker tuning:
  - `-w`, `--workers` (default: `9`)
  - `-t`, `--timeout` (default: `600`, mapped to `--timeout-worker-healthcheck`)
- Pet-server / arbiter:
  - `--num-pet-server` (default: `4`)
  - `--pet-server-start-port` (default: `8765`)
  - `--pet-server-cmd` (default: `pet-server`)
  - `--soft-max-ram-per-pet` (default: `4000` MB)
  - `--hard-max-ram-per-pet` (default: `6000` MB)
- Redis:
  - `--redis-port` (default: `6379`)
- Session/cache:
  - `--session-ttl-seconds` (default: `SESSION_TTL_SECONDS` env var, fallback `36000`)
  - `--session-cache-keep-feedback` (default: value from env `SESSION_CACHE_KEEP_FEEDBACK`, otherwise `false`)
  - `--session-cleanup-interval-seconds` (default: env `SESSION_CLEANUP_INTERVAL_SECONDS`, fallback `60`)
- Filesystem policy:
  - `--fs-access-mode` (`read_lib_only` or `rw_anywhere`; default from env `FS_ACCESS_MODE`, fallback `read_lib_only`)
  - `--coq-lib-path` (default from env `COQ_LIB_PATH`; otherwise resolved via `coqc -where`)
  - `--fs-read-allow` (repeatable extra read roots for `read_lib_only`)
- Compatibility placeholders:
  - `--app` (default: `rocq_ml_toolbox.inference.server:app`)
  - `--config` (default: `python:rocq_ml_toolbox.inference.gunicorn_config`)
  - these options are accepted for compatibility; current launcher always runs uvicorn with the built-in app target.

Environment notes:

- if `--fs-read-allow` is not passed, the launcher reads `FS_READ_ALLOW_PATHS` (JSON list) from env.
- CLI flags override env defaults for the corresponding options when both are provided.

Policy summary:
- `read_lib_only` (default): reads allowed under `coqc -where` plus any `--fs-read-allow` roots, writes denied.
- `rw_anywhere`: reads/writes allowed anywhere.
