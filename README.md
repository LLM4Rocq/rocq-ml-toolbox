# rocq-ml-toolbox

A toolbox for Rocq environments, scalable proof interaction, and project parsing for ML workflows.

## Components
- Docker: build and manage OPAM-based images, start Redis and the inference server in-container. See `src/rocq_ml_toolbox/docker/README.md`.
- Inference: a Flask/Gunicorn server that orchestrates pet-server workers with Redis caching, plus a Python client. See `src/rocq_ml_toolbox/inference/README.md`.
- Parser: proof and TOC extraction on top of the inference client. See `src/rocq_ml_toolbox/parser/README.md`.
- Rocq LSP: minimal LSP client for coq-lsp/rocq-lsp to fetch AST data. See `src/rocq_ml_toolbox/rocq_lsp/README.md`.

## Quick start (local, no Docker)
1) Install Python deps (plus pytanque):

```bash
pip install -e .[server,parser,client]
# Install pytanque separately and ensure `pet-server` is on PATH.
```

2) Start Redis and the inference server:

```bash
docker run -d -p 6379:6379 redis:latest
rocq-ml-server --num-pet-server 8 --workers 17
```

3) Use the Python client:

```python
from rocq_ml_toolbox.inference.client import PetClient

client = PetClient("127.0.0.1", 5000)
client.connect()
state = client.get_state_at_pos("/path/to/file.v", line=10, character=0)
state = client.run(state, "intros.")
goals = client.goals(state)
```

## Notebooks
- `notebooks/itp.ipynb`: interactive theorem proving at scale.
- `notebooks/scrapping.ipynb`: dataset extraction without Docker.
- `notebooks/docker.ipynb`: building and using custom Docker images.