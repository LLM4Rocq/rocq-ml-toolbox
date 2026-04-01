# Parser

Utilities for structured extraction from Rocq files on top of the inference service: AST, proof dumps, and replay-based dependency extraction.

## Requirements
- A running `rocq-ml-server`.
- Python deps via `pip install -e .[parser]` (and `pytanque` for server-side workers).

## Main API
- `RocqParser`: high-level parser bound to an inference client.
- `Source`: source wrapper with path/content helpers.
- `parser.extract_dump(...)`: fetch proof dump + AST + diagnostics.
- `parser.extract_proofs_raw(...)`: reconstruct proof scripts from AST nodes.
- `parser.extract_full_proof(...)`: replay proof steps and collect dependencies.

## Example
```python
from rocq_ml_toolbox.inference.client import PytanqueExtended
from rocq_ml_toolbox.parser.rocq_parser import RocqParser, Source

client = PytanqueExtended("127.0.0.1", 5000)
client.connect()
parser = RocqParser(client)

source = Source.from_local_path("/path/to/file.v")
proof_dump, ast, diags = parser.extract_dump(source, root="/path/to/project", force_dump=True)

print("proof entries:", len(proof_dump.proofs))
print("ast entries:", len(ast))
print("diagnostics:", len(diags))

for theorem_el, steps_raw in parser.extract_proofs_raw(source, ast)[:3]:
    theorem = parser.extract_full_proof(source, theorem_el, steps_raw, timeout=120)
    print(theorem.element.name, "steps:", len(theorem.steps))
```

## AST driver helpers
The `parser.ast.driver` module exposes:
- `load_ast_dump`, `parse_ast_dump` for JSONL AST dumps.
- `load_proof_dump`, `parse_proof_dump` for proofdeps dump payloads.
- `iter_v_files` for scanning `.v` files in a tree.
