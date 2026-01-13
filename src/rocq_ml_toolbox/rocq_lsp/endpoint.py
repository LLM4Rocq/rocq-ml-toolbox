"""
Extract from CoqPyt library (https://github.com/sr-lab/coqpyt)
Highly restricted version, only to get the AST from rocq-lsp.
"""

from typing import Dict, List

from .structs import Diagnostic, ResponseError
from .json_rpc_endpoint import JsonRpcEndpoint

class LspEndpoint:
    def __init__(self, json_rpc_endpoint, timeout=None):
        self.json_rpc_endpoint: JsonRpcEndpoint = json_rpc_endpoint
        self.next_id = 0
        self.timeout = timeout
        self.diagnostics: Dict[str, List[Diagnostic]] = {}

    def send_notification(self, method_name, **params):
        message = {
            "jsonrpc": "2.0",
            "method": method_name,
            "params": params,
        }
        self.json_rpc_endpoint.send_request(message)

    def call_method(self, method_name, **params):
        rpc_id = self.next_id
        self.next_id += 1

        message = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method_name,
            "params": params,
        }
        self.json_rpc_endpoint.send_request(message)

        while True:
            msg = self.json_rpc_endpoint.recv_response()
            if msg is None:
                raise RuntimeError("Server closed connection")

            if msg.get("id") == rpc_id:
                if "error" in msg:
                    err = msg["error"]
                    raise ResponseError(
                        err.get("code"),
                        err.get("message"),
                        err.get("data"),
                    )
                return msg.get("result")