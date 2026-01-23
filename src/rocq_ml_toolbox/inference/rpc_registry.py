# server/rpc_registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Optional
import inspect
from pytanque.protocol import Request  # union of param types

@dataclass
class RpcSpec:
    method: str
    fn: Callable[..., Any]  # bound method on SessionManager
    params_cls: type

class RpcRegistry:

    def __init__(self, client_cls):
        self.registry = {}
        for name, fn in inspect.getmembers(client_cls, predicate=inspect.isfunction):
            print(name)
            # method = getattr(fn, "__rpc_method__", None)
            # params_cls = getattr(fn, "__rpc_params_cls__", None)
            # if method and params_cls:
            #     self.registry[method] = RpcSpec(method=method, fn=fn, params_cls=params_cls)

class RpcError(Exception):
    def __init__(self, code: int, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message

    def to_error_obj(self):
        return {"code": self.code, "message": self.message}

def dispatch_rpc(registry: dict[str, RpcSpec], req: Request) -> Any:
    spec = registry.get(req.method_)
    if spec is None:
        raise RpcError(-32601, f"Method not found: {req.method}", http_status=404)
    params = spec.params_cls.from_json(req.params)
    return spec.fn(params=params)
