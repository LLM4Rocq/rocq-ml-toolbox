from __future__ import annotations
from typing import Dict, Tuple, List, Any, Optional, Set
import functools
from dataclasses import asdict, dataclass
import requests
import json

from pytanque.protocol import (
    Opts,
    State,
    Goal,
    Inspect,
    TocElement,
    GoalsResponse
)

from ..rocq_lsp.protocol import FlecheDocument
from ..parser.ast.driver import VernacElement, parse_ast_dump

@dataclass
class StateExtended(State):
    generation: int = 0
    
    @classmethod
    def from_json(cls, x: dict) -> StateExtended:
        base = State.from_json(x)
        gen = x.get("generation", 0)
        if not isinstance(gen, int):
            raise TypeError(f"generation must be int or None, got {type(gen)}")

        return cls(**asdict(base), generation=gen)
    
    def to_json(self) -> Any:
        base = super().to_json()
        base['generation'] = self.generation
        return base

    def to_state(self) -> State:
        return State(
            self.st,
            self.proof_finished,
            self.feedback,
            self.hash
        )
    
    @classmethod
    def from_state(cls, base: State, gen: int) -> StateExtended:
        return cls(**asdict(base), generation=gen)

class ClientError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

def retry(fn):
    """
    Retries the decorated method up to `retry` times (default 0) on ClientError
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        retry = int(kwargs.get("retry", 0) or 0)
        last_exc: Exception | None = None

        for _ in range(retry + 1):  # total attempts = 1 + retry
            try:
                return fn(*args, **kwargs)
            except ClientError as e:
                last_exc = e

        assert last_exc is not None
        raise last_exc
    return wrapper

def check_states(fn):
    """
    Ensure that all StateExtended arguments passed to a function are alive.
    """
    @functools.wraps(fn)
    def wrapper(self: PetClient, *args, **kwargs):
        for arg in args:
            if isinstance(arg, StateExtended):
                self._check_state(arg)

        for arg in kwargs.values():
            if isinstance(arg, StateExtended):
                self._check_state(arg)

        return fn(self, *args, **kwargs)
    return wrapper

class PetClient:
    """
    A simple client API for interacting with the pet Flask server.
    """
    def __init__(self, host: str, port: int):
        """
        Initialize the client by setting the base URL, logging in, and fetching theorem descriptions.
        """
        self.base_url = f"http://{host}:{port}"
        self.session_id: str = None
        self.alive_states: Set[StateExtended] = set() # keep track of alive intermediate states (run)
        self.dead_states: Set[StateExtended] = set() # keep track of dead intermediate states (each time get_state, or start is called, it killed all previous intermediate states)

    def connect(self):
        """
        Log in to the server to retrieve a load-balanced server index.
        """
        url = f"{self.base_url}/login"
        response = requests.get(url)
        if response.status_code == 200:
            output = response.json()
            self.session_id = output['session_id']
        else:
            raise ClientError(response.status_code, response.text)
    
    def _check_state(self, state: StateExtended):
        state_id = state.to_json_string()
        if state_id in self.dead_states:
            raise ClientError(400, 'The given state is dead, did you try to prove simultaneously multiple theorems?')

    def _reset_states(self):
        self.dead_states = self.alive_states | self.dead_states
        self.alive_states = set()
    
    def _add_state(self, state: StateExtended):
        state_id = state.to_json_string()
        self.alive_states.add(state_id)

    @retry
    def get_state_at_pos(self, filepath: str, line: int, character: int, opts: Optional[Opts]=None, failure: bool=False, timeout: int=120, retry: int=0) -> StateExtended:
        """
        Get state at position.
        """
        url = f"{self.base_url}/get_state_at_pos"
        if opts:
            opts = opts.to_json()
        payload = {'session_id': self.session_id, 'filepath': filepath, 'line': line, 'character': character, 'failure': failure, 'timeout': timeout, 'opts': opts}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            self._reset_states()
            return StateExtended.from_json(output['resp'])
        else:
            raise ClientError(response.status_code, response.text)

    @retry
    @check_states
    def run(self, state_ext: StateExtended, tactic: str, opts: Optional[Opts]=None, failure: bool=False, timeout: int=60, retry: int=0) -> StateExtended:
        """
        Execute a given tactic on the current proof state.
        """
        url = f"{self.base_url}/run"
        state_ext = state_ext.to_json()
        opts = opts.to_json() if opts else None
        payload = {'session_id': self.session_id, 'state_ext': state_ext, 'tactic': tactic, 'opts': opts, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            state_ext = StateExtended.from_json(output['resp'])
            self._add_state(state_ext)
            return state_ext
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    @check_states
    def goals(self, state_ext: StateExtended, pretty=True, failure: bool=False, timeout: int=10, retry: int=0) -> List[Goal]:
        """
        Gather goals associated to a state.
        """
        url = f"{self.base_url}/goals"
        state_ext = state_ext.to_json()
        payload = {'session_id': self.session_id, 'state_ext': state_ext, 'pretty': pretty, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return [Goal.from_json(goal) for goal in output['resp']]
        else:
            raise ClientError(response.status_code, response.text)

    @retry
    @check_states
    def complete_goals(self, state_ext: StateExtended, pretty=True, failure: bool=False, timeout: int=10, retry: int=0) -> Optional[GoalsResponse]:
        """
        Gather complete goals associated to a state.
        """
        url = f"{self.base_url}/complete_goals"
        state_ext = state_ext.to_json()
        payload = {'session_id': self.session_id, 'state_ext': state_ext, 'pretty': pretty, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            if not output['resp']:
                return None
            else:
                return GoalsResponse.from_json(output['resp'])
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    @check_states
    def premises(self, state_ext: StateExtended, failure: bool=False, timeout: int=10, retry: int=0) -> Any:
        """
        Gather accessible premises (lemmas, definitions) from a state.
        """
        url = f"{self.base_url}/premises"
        state_ext = state_ext.to_json()
        payload = {'session_id': self.session_id, 'state_ext': state_ext, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    @check_states
    def state_equal(self, st1_ext: StateExtended, st2_ext: StateExtended, kind=Inspect, failure: bool=False, timeout: int=10, retry: int=0) -> bool:
        """
        Check whether state st1 is equal to state st2.
        """
        url = f"{self.base_url}/state_equal"
        st1_ext = st1_ext.to_json()
        st2_ext = st2_ext.to_json()

        kind = kind.to_json()
        payload = {'session_id': self.session_id, 'st1_ext': st1_ext, 'st2_ext': st2_ext, 'kind': kind,'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    @check_states
    def state_hash(self, state_ext: StateExtended, failure: bool=False, timeout: int=10, retry: int=0) -> int:
        """
        Get a hash value for a proof state.
        """
        url = f"{self.base_url}/state_hash"
        state_ext = state_ext.to_json()

        payload = {'session_id': self.session_id, 'state_ext': state_ext, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def toc(self, file: str, failure: bool=False, timeout: int=120, retry: int=0) -> List[Tuple[str, List[TocElement]]]:
        """
        Get toc of a file.
        """
        url = f"{self.base_url}/toc"
        payload = {'session_id': self.session_id, 'file': file, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            typed_output = [(x[0],[TocElement.from_json(y) for y in x[1]]) for x in output['resp']]
            return typed_output
        else:
            raise ClientError(response.status_code, response.text)

    def get_document(self, path: str) -> FlecheDocument:
        """
        Get fleche representation of document at path `path`.
        """
        url = f"{self.base_url}/get_document"
        payload = {'path': path}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return FlecheDocument.from_json(output)
        else:
            raise ClientError(response.status_code, response.text)

    @retry
    def get_ast(self, path: str, force_dump=False, retry: int=0) -> List[VernacElement]:
        """
        Get AST of document at path `path`.
        If `force_dump` is set to `True`, then force AST recomputing (may be necessary if the first time the AST was only partially generated.
        """
        url = f"{self.base_url}/get_ast"
        payload = {'path': path, 'force_dump': force_dump}
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                raise ClientError(response.status_code, response.text)

            tmp_path = "/tmp/ast_dump.json"
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        data = []
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return parse_ast_dump(data)

    @retry
    @check_states
    def ast(self, state_ext: StateExtended, text: str, failure: bool=False, timeout: int=10, retry: int=0) -> Dict:
        """
        Get ast of a command parsed at a state.
        """
        url = f"{self.base_url}/ast"
        state_ext = state_ext.to_json()
        payload = {'session_id': self.session_id, 'state_ext': state_ext, 'text': text, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def ast_at_pos(self, file: str, line: int, character: int, failure: bool=False, timeout: int=10, retry: int=0) -> Dict:
        """
        Get ast at a specified position in a file.
        """
        url = f"{self.base_url}/ast_at_pos"
        payload = {'session_id': self.session_id, 'file': file, 'line': line, 'character': character, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def get_root_state(self, file: str, opts: Optional[Opts]=None, failure: bool=False, timeout: int=10, retry: int=0) -> StateExtended:
        """
        Get root state of a document.
        """
        url = f"{self.base_url}/get_root_state"
        opts = opts.to_json() if opts else None
        payload = {'session_id': self.session_id, 'file': file, 'opts': opts, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return StateExtended.from_json(output['resp'])
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    @check_states
    def list_notations_in_statement(self, state_ext: StateExtended, statement: str, failure: bool=False, timeout: int=10, retry: int=0) -> list[Dict]:
        """
        Get the list of notations appearing in a theorem/lemma statement.
        """
        url = f"{self.base_url}/list_notations_in_statement"
        state_ext = state_ext.to_json()
        payload = {'session_id': self.session_id, 'state_ext': state_ext, 'statement': statement, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def start(self, file: str, thm: str, pre_commands: Optional[str]=None, opts: Optional[Opts]=None, failure: bool=False, timeout: int=10, retry: int=0) -> StateExtended:
        """
        Start a proof session for a specific theorem in a Coq/Rocq file.
        """
        url = f"{self.base_url}/start"
        opts = opts.to_json() if opts else None
        payload = {'session_id': self.session_id, 'file': file, 'thm': thm, 'pre_commands': pre_commands, 'opts': opts, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            self._reset_states()
            return StateExtended.from_json(output['resp'])
        else:
            raise ClientError(response.status_code, response.text)

    # @retry
    # def query(self, params: Params, size: int=4096, failure: bool=False, timeout: int=10) -> Response:
    #     """
    #     Send a low-level JSON-RPC query to the server.
    #     """
    #     url = f"{self.base_url}/query"
    #     params = params.to_json()
    #     payload = {'session_id': self.session_id, 'params': params, 'size': size, 'failure': failure, 'timeout': timeout}
    #     response = requests.post(url, json=payload)
    #     if response.status_code == 200:
    #         output = response.json()
    #         return Response.from_json(output['resp'])
    #     else:
    #         raise ClientError(response.status_code, response.text)

    def get_session(self) -> Dict[str, Any]:
        url = f"{self.base_url}/get_session"
        payload = {'session_id': self.session_id}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output
        else:
            raise ClientError(response.status_code, response.text)