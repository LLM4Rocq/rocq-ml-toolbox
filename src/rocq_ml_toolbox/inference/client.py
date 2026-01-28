from pytanque import Pytanque, PytanqueMode
import requests

from ..parser.ast.driver import parse_ast_dump

class PytanqueExtended(Pytanque):
    def __init__(self, host, port):
        super().__init__(host=host, port=port, mode=PytanqueMode.HTTP)

    def get_ast(self, path: str, force_dump:bool=False):
        url = f"http://{self.host}:{self.port}/get_ast"
        result = requests.post(url, json={"path":path, "force_dump": force_dump})
        raw_ast = result.json()['value']
        return parse_ast_dump(raw_ast)