# rocq-ml-toolbox

Toolbox for ML workflows with Rocq/Coq: scalable proof interaction, parser utilities, Docker helpers, and safe proof-checking.

## Components
- Inference server (`rocq-ml-server`): FastAPI + uvicorn API with Redis-backed sessions and an arbiter supervising multiple `pet-server` workers. See `src/rocq_ml_toolbox/inference/README.md`.
- Parser: proof/AST extraction utilities layered on top of the inference client. See `src/rocq_ml_toolbox/parser/README.md`.
- SafeVerify: verify that a target file safely discharges obligations from a source file. See `examples/safeverify/README.md`.
- Docker helpers: build OPAM-based images and run the inference stack in-container. See `src/rocq_ml_toolbox/docker/README.md`.
- Rocq LSP: minimal JSON-RPC client for `coq-lsp` AST access. See `src/rocq_ml_toolbox/rocq_lsp/README.md`.

## Quick Start (Local)
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
For dump/dependency routes (`/get_dump`), you also need `fcc` + patched `rocq-lsp` plugin support from:
- https://github.com/theostos/rocq-lsp
- use branch `proofdepsdump-dump-load-v9.1` / `proofdepsdump-dump-load-v9.0` / `proofdepsdump-dump-load-v8.20` depending on your Coq/Rocq version.

3) Use the Python client:

```python
from rocq_ml_toolbox.inference.client import PytanqueExtended

client = PytanqueExtended("127.0.0.1", 5000)
client.connect()

state = client.get_state_at_pos("/path/to/file.v", line=10, character=0)
state = client.run(state, "intros.")
print(client.goals(state))
```

## Quick Start (Docker)

Use a prebuilt image (for example `theostos/coq-mathcomp:9.0-2.5.0`) and expose API port `5000`:

```bash
docker run --rm -p 5000:5000 theostos/coq-mathcomp:9.0-2.5.0 rocq-ml-server --host 0.0.0.0 \
--port 5000 \
--num-pet-server 2 \
--workers 3
```

Then from host:

```python
from rocq_ml_toolbox.inference.client import PytanqueExtended

client = PytanqueExtended("127.0.0.1", 5000)
client.connect()
```

For Coq 8.20 images, use `theostos/coq-server:8.20` and `/home/coq/miniconda/envs/rocq-ml/bin`.

## `rocq-ml-server` CLI

Full CLI argument reference (defaults, semantics, env-var behavior) is documented in:

- `src/rocq_ml_toolbox/inference/README.md` (`CLI Reference` section)

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
