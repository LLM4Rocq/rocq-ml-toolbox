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

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        url = f"http://{self.host}:{self.port}/{endpoint.lstrip('/')}"
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _ensure_dict(payload: Any, *, endpoint: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid response from {endpoint}: expected object, got {type(payload).__name__}")
        return payload

    def get_dump(self, path: Union[Path, str], root: Optional[Union[Path, str]]=None, force_dump: bool=True) -> Tuple[ProofDump, List[VernacElement], List[Diagnostic]]:
        if root:
            root = str(root)
        result = self._ensure_dict(
            self._post_json("get_dump", {"path": str(path), "root": root, "force_dump": force_dump}),
            endpoint="/get_dump",
        )
        raw_proof = result['proof']
        raw_ast = result['ast']
        diags = result['diags']
        return ProofDump.from_json(raw_proof), parse_ast_dump(raw_ast), [Diagnostic.from_json(d) for d in diags]
    
    def get_glob(self, path: Union[Path, str], force_compile:bool=False) -> GlobFile:
        result = self._ensure_dict(
            self._post_json("get_glob", {"path": str(path), "force_compile": force_compile}),
            endpoint="/get_glob",
        )
        if "value" not in result:
            raise ValueError("Invalid response from /get_glob: missing `value`.")
        glob_file = GlobFile.from_json(result['value'])
        return glob_file
    
    def tmp_file(self, content: Optional[str]=None, root: Optional[Union[Path, str]]=None) -> str:
        payload = {
            "content": content,
            "root": None if root is None else str(root),
        }
        result = self._ensure_dict(self._post_json("tmp_file", payload), endpoint="/tmp_file")
        path = result.get('path')
        if not isinstance(path, str):
            raise ValueError("Invalid response from /tmp_file: missing string `path`.")
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
        payload = {
            "source": str(source),
            "target": str(target),
            "root": str(root),
            "axiom_whitelist": list(axiom_whitelist or []),
            "save_path": None if save_path is None else str(save_path),
            "verbose": verbose,
        }
        result = self._ensure_dict(self._post_json("safeverify", payload), endpoint="/safeverify")
        return result

    def access_libraries(
        self,
        env: str | None = None,
        *,
        use_cache: bool = True,
        include_theories: bool = True,
        include_user_contrib: bool = True,
    ) -> dict[str, Any]:
        result = self._ensure_dict(
            self._post_json(
                "access_libraries",
                {
                    "env": env,
                    "use_cache": use_cache,
                    "include_theories": include_theories,
                    "include_user_contrib": include_user_contrib,
                },
            ),
            endpoint="/access_libraries",
        )
        if not isinstance(result.get("nodes"), list):
            raise ValueError("Invalid response from /access_libraries: missing list `nodes`.")
        if not isinstance(result.get("file_index"), dict):
            raise ValueError("Invalid response from /access_libraries: missing object `file_index`.")
        return result

    def read_file(
        self,
        path: Union[Path, str],
        *,
        offset: int = 0,
        max_chars: int = 20000,
    ) -> dict[str, Any]:
        result = self._ensure_dict(
            self._post_json(
                "read_file",
                {
                    "path": str(path),
                    "offset": offset,
                    "max_chars": max_chars,
                },
            ),
            endpoint="/read_file",
        )
        content = result.get("content")
        if not isinstance(content, str):
            raise ValueError("Invalid response from /read_file: missing string `content`.")
        next_offset = result.get("next_offset")
        eof = result.get("eof")
        if not isinstance(next_offset, int) or not isinstance(eof, bool):
            raise ValueError("Invalid response from /read_file: missing `next_offset` or `eof`.")
        return result

    def write_file(
        self,
        path: Union[Path, str],
        *,
        content: str,
        offset: int = 0,
        truncate: bool = False,
    ) -> dict[str, Any]:
        result = self._ensure_dict(
            self._post_json(
                "write_file",
                {
                    "path": str(path),
                    "content": content,
                    "offset": offset,
                    "truncate": truncate,
                },
            ),
            endpoint="/write_file",
        )
        bytes_written = result.get("bytes_written")
        if not isinstance(bytes_written, int):
            raise ValueError("Invalid response from /write_file: missing integer `bytes_written`.")
        return result

    def read_docstrings(self, source: Union[Path, str]) -> list[dict[str, Any]]:
        result = self._ensure_dict(
            self._post_json("read_docstrings", {"source": str(source)}),
            endpoint="/read_docstrings",
        )
        docstrings = result.get("docstrings")
        if not isinstance(docstrings, list):
            raise ValueError("Invalid response from /read_docstrings: missing list `docstrings`.")
        validated: list[dict[str, Any]] = []
        for item in docstrings:
            if isinstance(item, dict):
                validated.append(item)
        return validated
    
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
