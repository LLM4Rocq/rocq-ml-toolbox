"""
Extract from CoqPyt library (https://github.com/sr-lab/coqpyt)
Highly restricted version, only to get the AST from rocq-lsp.
"""

from typing import Optional
import subprocess

from .endpoint import LspEndpoint
from .structs import TextDocumentItem, TextDocumentIdentifier
from .protocol import FlecheDocument
from .json_rpc_endpoint import JsonRpcEndpoint

class LspClient():
    def __init__(self):
        """
        Start a coq-lsp subprocess and initialize the JSON-RPC/LSP endpoints.
        """
        self.proc = subprocess.Popen(
            "coq-lsp -D 0",
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            shell=True,
        )
        json_rpc_endpoint = JsonRpcEndpoint(self.proc.stdin, self.proc.stdout)
        self.lsp_endpoint = LspEndpoint(json_rpc_endpoint, timeout=10)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        """Terminate the coq-lsp subprocess."""
        if self.proc is None:
            return

        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                self.proc.kill()
                self.proc.wait()

        self.proc = None

    def initialize(
        self,
        item: TextDocumentItem
    ):
        """
        Send the LSP initialize request.
        Must be called exactly once before any other LSP requests
        """

        __DEFAULT_INIT_OPTIONS = {
            "max_errors": 120000000,
            "goal_after_tactic": False,
            "show_coq_info_messages": True,
            "eager_diagnostics": False
        }
        workspaces = [{"name": "coq-lsp", "uri": item.uri}]

        return self.lsp_endpoint.call_method(
            "initialize",
            processId=self.proc.pid,
            rootPath="",
            rootUri=item.uri,
            initializationOptions=__DEFAULT_INIT_OPTIONS,
            capabilities={},
            trace="off",
            workspaceFolders=workspaces,
        )

    def didOpen(self, textDocument: TextDocumentItem):
        """
        Notify the server that a text document is opened and managed by the client.
        """
        return self.lsp_endpoint.send_notification(
            "textDocument/didOpen", textDocument=textDocument
        )


    def getDocument(
        self, textDocument: TextDocumentIdentifier
    ) -> Optional[FlecheDocument]:
        """
        Retrieve the AST for an opened text document.
        """
        result_dict = self.lsp_endpoint.call_method(
            "coq/getDocument", textDocument=textDocument
        )
        if result_dict:
            return FlecheDocument.from_json(result_dict)
        else:
            return None