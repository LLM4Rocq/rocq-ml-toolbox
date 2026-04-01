from pytanque import Pytanque, PytanqueMode
import requests
from pathlib import Path
from typing import List, Optional, Union, Any, Self, Tuple, Sequence

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
    
    def tmp_file(self, content: Optional[str]=None) -> str:
        url = f"http://{self.host}:{self.port}/tmp_file"
        result = requests.post(url, {"content": content})
        path = result.json()['path']
        return path

    def safeverify(
        self,
        source: Union[Path, str],
        target: Union[Path, str],
        root: Union[Path, str],
        axiom_whitelist: Optional[Sequence[str]] = None,
        save_path: Optional[Union[Path, str]] = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        url = f"http://{self.host}:{self.port}/safeverify"
        payload = {
            "source": str(source),
            "target": str(target),
            "root": str(root),
            "axiom_whitelist": list(axiom_whitelist or []),
            "save_path": None if save_path is None else str(save_path),
            "verbose": verbose,
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
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
