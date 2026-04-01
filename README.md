# rocq-ml-toolbox

Toolbox for ML workflows with Rocq/Coq: scalable proof interaction, parser utilities, Docker helpers, and safe proof-checking.

## Components
- Inference server (`rocq-ml-server`): FastAPI + uvicorn API with Redis-backed sessions and an arbiter supervising multiple `pet-server` workers. See `src/rocq_ml_toolbox/inference/README.md`.
- Parser: proof/AST extraction utilities layered on top of the inference client. See `src/rocq_ml_toolbox/parser/README.md`.
- SafeVerify: verify that a target file safely discharges obligations from a source file. See `examples/safeverify/README.md`.
- Docker helpers: build OPAM-based images and run the inference stack in-container. See `src/rocq_ml_toolbox/docker/README.md`.
- Rocq LSP: minimal JSON-RPC client for `coq-lsp` AST access. See `src/rocq_ml_toolbox/rocq_lsp/README.md`.

## Quick Start (Local, No Docker)
1) Install package dependencies:

```bash
pip install -e .[all]
pip install git+https://github.com/llm4rocq/pytanque.git
```

2) Start the server:

```bash
rocq-ml-server --num-pet-server 4 --workers 9 --port 5000
```

`rocq-ml-server` starts `redis-server`, the arbiter, and uvicorn. Ensure `redis-server` and `pet-server` are available on `PATH`.

3) Use the Python client:

```python
from rocq_ml_toolbox.inference.client import PytanqueExtended

client = PytanqueExtended("127.0.0.1", 5000)
client.connect()

state = client.get_state_at_pos("/path/to/file.v", line=10, character=0)
state = client.run(state, "intros.")
print(client.goals(state))
```

## SafeVerify Quick Example
```bash
rocq-ml-safeverify \
  examples/safeverify/nontrivial/Source.v \
  examples/safeverify/nontrivial/TargetGood.v \
  --root examples/safeverify -v
```

The same check is available through the inference API via `client.safeverify(...)`.

## Notebooks
- `notebooks/itp.ipynb`: interactive theorem proving at scale.
- `notebooks/scrapping.ipynb`: data extraction without Docker.
- `notebooks/docker.ipynb`: building and using custom Docker images.
