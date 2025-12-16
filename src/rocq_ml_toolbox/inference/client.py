from typing import Dict, Tuple, List, Any, Optional

import requests

from pytanque.client import Params, Response
from pytanque.protocol import (
    Response,
    Opts,
    State,
    Goal,
    Inspect
)

class ClientError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

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
        self._login()

    def _login(self):
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
    
    def get_state_at_pos(self, filepath: str, line: int, character: int, failure: bool=False, timeout: int=120) -> State:
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

    def run(self, state: State, tactic: str, failure: bool=False, timeout: int=60) -> State:
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
    
    def goals(self, state: State, pretty=True, failure: bool=False, timeout: int=10) -> List[Goal]:
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

    def complete_goals(self, state: State, pretty=True, failure: bool=False, timeout: int=10) -> Dict:
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
    
    def premises(self, state: State, failure: bool=False, timeout: int=10) -> Any:
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
    
    def state_equal(self, st1: State, st2: State, kind=Inspect, failure: bool=False, timeout: int=10) -> bool:
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
    
    def state_hash(self, state: State, failure: bool=False, timeout: int=10) -> int:
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
    
    def toc(self, file: str, failure: bool=False, timeout: int=10) -> list[tuple[str, Any]]:
        """
        Get toc of a file.
        """
        url = f"{self.base_url}/toc"
        payload = {'session_id': self.session_id, 'file': file, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['resp']
        else:
            raise ClientError(response.status_code, response.text)
    
    def ast(self, state: State, text: str, failure: bool=False, timeout: int=10) -> Dict:
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
    
    def ast_at_pos(self, file: str, line: int, character: int, failure: bool=False, timeout: int=10) -> Dict:
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
    
    def get_root_state(self, file: str, opts: Optional[Opts]=None, failure: bool=False, timeout: int=10) -> State:
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
    
    def list_notations_in_statement(self, state: State, statement: str, failure: bool=False, timeout: int=10) -> list[Dict]:
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
    
    def start(self, file: str, thm: str, pre_commands: Optional[str]=None, opts: Optional[Opts]=None, failure: bool=False, timeout: int=10) -> State:
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