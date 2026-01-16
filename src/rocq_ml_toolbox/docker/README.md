# Docker

Tools to build and run OPAM-based Rocq/Coq environments with Docker. Includes helpers to start Redis, run the inference server in-container, and extract `.v` sources.

## Base images
The `docker/docker-compose.yaml` file builds four images:
- `theostos/rocq-lsp:9.0`, `theostos/rocq-server:9.0`
- `theostos/coq-lsp:8.20`, `theostos/coq-server:8.20`

Build them with:

```bash
docker compose -f docker/docker-compose.yaml build
```

## OpamConfig YAML
`OpamConfig.from_yaml` expects:
- `name`, `tag`, `base_image`, `user`
- `opam_env_path`
- `packages` (list)
- `pins` (optional list)

See `notebooks/coq-corn.yaml` for a minimal example.

## Example

```python
from rocq_ml_toolbox.docker import OpamConfig, OpamDocker

cfg = OpamConfig.from_yaml("notebooks/coq-corn.yaml")
opam_docker = OpamDocker(cfg)
try:
    opam_docker.start_inference_server(port=5000)
    sources = opam_docker.extract_source_files_from_folder("Stdlib")
    print(len(sources))
finally:
    opam_docker.close()
```

## Useful helpers
- `list_opam_folder`, `extract_source_files_from_folder`, `extract_source_files_from_package`.
- `get_source` and `upload_source` to read/write `.v` files inside the container.
