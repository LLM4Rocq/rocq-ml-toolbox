from pytanque import Pytanque, PytanqueMode
import requests
from pathlib import Path
from typing import List, Optional, Union, Any, Self

from ..parser.ast.driver import parse_ast_dump, VernacElement
from ..parser.glob.driver import GlobFile
class PytanqueExtended(Pytanque):
    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port, mode=PytanqueMode.HTTP)

    def get_ast(self, path: Union[Path, str], root: Optional[Union[Path, str]]=None, force_dump: bool=False) -> List[VernacElement]:
        url = f"http://{self.host}:{self.port}/get_ast"
        result = requests.post(url, json={"path": str(path), "root": str(root), "force_dump": force_dump})
        raw_ast = result.json()['value']
        return parse_ast_dump(raw_ast)
    
    def get_glob(self, path: Union[Path, str], force_compile:bool=False) -> GlobFile:
        url = f"http://{self.host}:{self.port}/get_glob"
        result = requests.post(url, json={"path": str(path), "force_compile": force_compile})
        glob_file = GlobFile.from_json(result.json()['value'])
        return glob_file
    
    def empty_file(self) -> str:
        url = f"http://{self.host}:{self.port}/empty_file"
        result = requests.get(url)
        path = result.json()['path']
        return path
    
    def to_json(self) -> Any:
        return {
            "host": self.host,
            "port": self.port,
            "session_id": self.session_id
        }

    @classmethod
    def from_json(cls, x) -> Self:
        client = cls(
            x['host'],
            x['port']
        )
        client.session_id = x['session_id']
        return client