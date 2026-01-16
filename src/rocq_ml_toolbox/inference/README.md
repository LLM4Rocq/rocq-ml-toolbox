# Inference

Flask + Gunicorn service that manages multiple pet-server workers through Redis, with a Python client on top of the HTTP API.

## Requirements
- Redis server.
- `pet-server` from pytanque on PATH.
- Python deps via `pip install -e .[server]`.
- `Rocq Prover` and `coq-lsp`.

## Run the server

```bash
rocq-ml-server --num-pet-server 4 --workers 9 --port 5000
```

The CLI sets the environment variables consumed by `gunicorn_config.py`.

## Python client

```python
from rocq_ml_toolbox.inference.client import PetClient

client = PetClient("127.0.0.1", 5000)
client.connect()

state = client.get_state_at_pos("/path/to/file.v", line=10, character=0)
state = client.run(state, "intros.")
print(client.goals(state))
```

Client methods accept `timeout` and `retry` to handle slow or flaky pet-server calls.

## API surface
- `GET /health`: check pet-server readiness.
- `GET /login`: create a session.
- `POST /get_state_at_pos`, `/get_root_state`, `/start`: state acquisition.
- `POST /run`, `/goals`, `/complete_goals`, `/premises`: proof interaction.
- `POST /state_equal`, `/state_hash`: state comparison.
- `POST /toc`, `/ast`, `/ast_at_pos`: parsing helpers.
- `POST /list_notations_in_statement`: notation extraction.
- `POST /get_document`, `/get_ast`: LSP or AST dump extraction.
- `POST /get_session`: session metadata.

## Configuration knobs
- `NUM_PET_SERVER`, `PET_SERVER_START_PORT`: worker pool size and base port.
- `MAX_RAM_PER_PET`: restart a pet-server when it exceeds this MB limit.
- `REDIS_URL`: Redis endpoint.
- `--detached` plus `--pidfile`, `--errorlog`, `--accesslog` for daemon mode.
