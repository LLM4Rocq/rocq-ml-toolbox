from typing import Dict, Tuple, List, Any, Optional
import functools

import requests

from pytanque.protocol import (
    Opts,
    State,
    Goal,
    Inspect,
    TocElement
)


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

class PetClient:
    """
    A simple client API for interacting with the pet Flask server.
    """
    def __init__(self, base_url: str):
        """
        Initialize the client by setting the base URL, logging in, and fetching theorem descriptions.
        """
        self.base_url = base_url.rstrip("/")
        self.session_id: str = None

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
    
    @retry
    def get_state_at_pos(self, filepath: str, line: int, character: int, failure: bool=False, timeout: int=120, retry: int=0) -> State:
        """
        Get state at position.
        """
        url = f"{self.base_url}/get_state_at_pos"
        payload = {'session_id': self.session_id, 'filepath': filepath, 'line': line, 'character': character, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return State.from_json(output['resp'])
        else:
            raise ClientError(response.status_code, response.text)

    @retry
    def run(self, state: State, tactic: str, failure: bool=False, timeout: int=60, retry: int=0) -> State:
        """
        Execute a given tactic on the current proof state.
        """
        url = f"{self.base_url}/run"
        state = state.to_json()
        payload = {'session_id': self.session_id, 'state': state, 'tactic': tactic, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return State.from_json(output['resp'])
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def goals(self, state: State, pretty=True, failure: bool=False, timeout: int=10, retry: int=0) -> List[Goal]:
        """
        Gather goals associated to a state.
        """
        url = f"{self.base_url}/goals"
        state = state.to_json()
        payload = {'session_id': self.session_id, 'state': state, 'pretty': pretty, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return [Goal.from_json(goal) for goal in output['resp']]
        else:
            raise ClientError(response.status_code, response.text)

    @retry
    def complete_goals(self, state: State, pretty=True, failure: bool=False, timeout: int=10, retry: int=0) -> Dict:
        """
        Gather complete goals associated to a state.
        """
        url = f"{self.base_url}/complete_goals"
        state = state.to_json()
        payload = {'session_id': self.session_id, 'state': state, 'pretty': pretty, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def premises(self, state: State, failure: bool=False, timeout: int=10, retry: int=0) -> Any:
        """
        Gather accessible premises (lemmas, definitions) from a state.
        """
        url = f"{self.base_url}/premises"
        state = state.to_json()
        payload = {'session_id': self.session_id, 'state': state, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def state_equal(self, st1: State, st2: State, kind=Inspect, failure: bool=False, timeout: int=10, retry: int=0) -> bool:
        """
        Check whether state st1 is equal to state st2.
        """
        url = f"{self.base_url}/state_equal"
        st1 = st1.to_json()
        st2 = st2.to_json()
        kind = kind.to_json()
        payload = {'session_id': self.session_id, 'st1': st1, 'st2': st2, 'kind': kind,'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def state_hash(self, state: State, failure: bool=False, timeout: int=10, retry: int=0) -> int:
        """
        Get a hash value for a proof state.
        """
        url = f"{self.base_url}/state_hash"
        state = state.to_json()
        payload = {'session_id': self.session_id, 'state': state, 'failure': failure, 'timeout': timeout}
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
    
    @retry
    def ast(self, state: State, text: str, failure: bool=False, timeout: int=10, retry: int=0) -> Dict:
        """
        Get ast of a command parsed at a state.
        """
        url = f"{self.base_url}/ast"
        state = state.to_json()
        payload = {'session_id': self.session_id, 'state': state, 'text': text, 'failure': failure, 'timeout': timeout}
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
    def get_root_state(self, file: str, opts: Optional[Opts]=None, failure: bool=False, timeout: int=10, retry: int=0) -> State:
        """
        Get root state of a document.
        """
        url = f"{self.base_url}/get_root_state"
        opts = opts.to_json() if opts else None
        payload = {'session_id': self.session_id, 'file': file, 'opts': opts, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return State.from_json(output['resp'])
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def list_notations_in_statement(self, state: State, statement: str, failure: bool=False, timeout: int=10, retry: int=0) -> list[Dict]:
        """
        Get the list of notations appearing in a theorem/lemma statement.
        """
        url = f"{self.base_url}/list_notations_in_statement"
        state = state.to_json()
        payload = {'session_id': self.session_id, 'state': state, 'statement': statement, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    @retry
    def start(self, file: str, thm: str, pre_commands: Optional[str]=None, opts: Optional[Opts]=None, failure: bool=False, timeout: int=10, retry: int=0) -> State:
        """
        Start a proof session for a specific theorem in a Coq/Rocq file.
        """
        url = f"{self.base_url}/start"
        opts = opts.to_json() if opts else None
        payload = {'session_id': self.session_id, 'file': file, 'thm': thm, 'pre_commands': pre_commands, 'opts': opts, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return State.from_json(output['resp'])
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