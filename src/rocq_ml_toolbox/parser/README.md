# Parser

Utilities to extract structured data from Rocq projects: TOC, proofs, proof steps, and dependencies. Built on top of the inference client.

## Requirements
- A running `rocq-ml-server` instance.
- Python deps via `pip install -e .[parser]` (plus `pytanque` for the server).

## Main API
- `RocqParser`: orchestrates parsing via a `PetClient`.
- `Source`: file content wrapper with UTF-8 helpers.
- `Theorem`, `Step`, `Dependency`: structured proof data.

## Example

```python
from rocq_ml_toolbox.inference.client import PetClient
from rocq_ml_toolbox.parser.rocq_parser import RocqParser, Source

client = PetClient("127.0.0.1", 5000)
client.connect()
parser = RocqParser(client)

source = Source.from_local_path("/path/to/file.v")
for proof in parser.extract_proofs(source):
    print(proof.element.name)
    for step in proof.steps:
        print(step.step)
```

## AST helpers
The `parser.ast.driver` module exposes:
- `load_ast_dump` and `parse_ast_dump` for JSONL dumps.
- `compute_ast` for a one-shot parse.
- `iter_v_files` to iterate over `.v` files in a tree.
