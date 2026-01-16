# Rocq LSP

A minimal JSON-RPC/LSP client for `coq-lsp` used to fetch `FlecheDocument` AST data. This is a focused, read-only subset of the LSP protocol.

## Example

```python
from rocq_ml_toolbox.rocq_lsp.client import LspClient
from rocq_ml_toolbox.rocq_lsp.structs import TextDocumentItem

with LspClient() as client:
    item = TextDocumentItem("/path/to/file.v")
    client.initialize(item)
    client.didOpen(item)
    doc = client.getDocument(item)
    print(len(doc.spans))
```

## Notes
- `protocol.py` is generated from `protocol.atd` and provides typed JSON helpers.
- The inference server uses this client for `/get_document`.
