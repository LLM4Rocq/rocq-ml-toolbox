from pytanque import Pytanque, PytanqueMode
import requests
from pathlib import Path
from typing import List, Optional, Union, Any, Self, Tuple

from ..parser.diags.parser import Diagnostic
from ..parser.ast.driver import parse_ast_dump, VernacElement
from ..parser.glob.driver import GlobFile
from ..parser.proof.parser import ProofDump

class PytanqueExtended(Pytanque):
    def __init__(self, host: str, port: int):
        super().__init__(host=host, port=port, mode=PytanqueMode.HTTP)

    def get_dump(self, path: Union[Path, str], root: Optional[Union[Path, str]]=None, force_dump: bool=True) -> Tuple[ProofDump, List[VernacElement], List[Diagnostic]]:
        url = f"http://{self.host}:{self.port}/get_dump"
        if root:
            root = str(root)
        result = requests.post(url, json={"path": str(path), "root": root, "force_dump": force_dump}).json()
        raw_proof = result['proof']
        raw_ast = result['ast']
        diags = result['diags']
        return ProofDump.from_json(raw_proof), parse_ast_dump(raw_ast), [Diagnostic.from_json(d) for d in diags]
    
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