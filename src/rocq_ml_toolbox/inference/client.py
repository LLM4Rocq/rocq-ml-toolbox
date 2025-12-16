from typing import Dict, Tuple, List, Any, Optional

import requests

State = Dict[str, str]
Goals = List[Dict[str, str]]

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
            self.session_id = response.json()['session_id']
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
            return output['state']
        else:
            raise ClientError(response.status_code, response.text)

    def run(self, state: State, tactic: str, failure: bool=False, timeout: int=60) -> State:
        """
        Execute a given tactic on the current proof state.
        """
        url = f"{self.base_url}/run"
        payload = {'session_id': self.session_id, 'state': state, 'tactic': tactic, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['state']
        else:
            raise ClientError(response.status_code, response.text)
    
    def goals(self, state: State, pretty=True, failure: bool=False, timeout: int=10) -> Goals:
        """
        Gather goals associated to a state.
        """
        url = f"{self.base_url}/goals"
        payload = {'session_id': self.session_id, 'state': state, 'pretty': pretty, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['goals']
        else:
            raise ClientError(response.status_code, response.text)

    def complete_goals(self, state: State, pretty=True, failure: bool=False, timeout: int=10) -> Goals:
        """
        Gather complete goals associated to a state.
        """
        url = f"{self.base_url}/complete_goals"
        payload = {'session_id': self.session_id, 'state': state, 'pretty': pretty, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['goals']
        else:
            raise ClientError(response.status_code, response.text)
    
    def premises(self, state: State, failure: bool=False, timeout: int=10) -> Goals:
        """
        Gather accessible premises (lemmas, definitions) from a state.
        """
        url = f"{self.base_url}/premises"
        payload = {'session_id': self.session_id, 'state': state, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['premises']
        else:
            raise ClientError(response.status_code, response.text)
    
    def state_equal(self, st1: State, st2: State, kind=None, failure: bool=False, timeout: int=10) -> bool:
        """
        Check whether state st1 is equal to state st2.
        """
        url = f"{self.base_url}/state_equal"
        payload = {'session_id': self.session_id, 'st1': st1, 'st2': st2, 'kind': kind,'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['result']
        else:
            raise ClientError(response.status_code, response.text)
    
    def state_hash(self, state: State, failure: bool=False, timeout: int=10) -> Goals:
        """
        Get a hash value for a proof state.
        """
        url = f"{self.base_url}/state_hash"
        payload = {'session_id': self.session_id, 'state': state, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['hash']
        else:
            raise ClientError(response.status_code, response.text)
    
    def toc(self, file: str, failure: bool=False, timeout: int=10) -> Goals:
        """
        Get toc of a file.
        """
        url = f"{self.base_url}/toc"
        payload = {'session_id': self.session_id, 'file': file, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['toc']
        else:
            raise ClientError(response.status_code, response.text)
    
    def ast(self, state: State, text: str, failure: bool=False, timeout: int=10) -> Goals:
        """
        Get ast of a command parsed at a state.
        """
        url = f"{self.base_url}/ast"
        payload = {'session_id': self.session_id, 'state': state, 'text': text, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['ast']
        else:
            raise ClientError(response.status_code, response.text)
    
    def ast_at_pos(self, file: str, line: int, character: int, failure: bool=False, timeout: int=10) -> Goals:
        """
        Get ast at a specified position in a file.
        """
        url = f"{self.base_url}/ast_at_pos"
        payload = {'session_id': self.session_id, 'file': file, 'line': line, 'character': character, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['ast']
        else:
            raise ClientError(response.status_code, response.text)
    
    def get_root_state(self, file: str, failure: bool=False, timeout: int=10) -> Goals:
        """
        Get root state of a document.
        """
        url = f"{self.base_url}/get_root_state"
        payload = {'session_id': self.session_id, 'file': file, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['state']
        else:
            raise ClientError(response.status_code, response.text)
    
    def list_notations_in_statement(self, state: State, statement: str, failure: bool=False, timeout: int=10) -> Goals:
        """
        Get the list of notations appearing in a theorem/lemma statement.
        """
        url = f"{self.base_url}/list_notations_in_statement"
        payload = {'session_id': self.session_id, 'state': state, 'statement': statement, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['notations']
        else:
            raise ClientError(response.status_code, response.text)
    
    def start(self, file: str, thm: str, failure: bool=False, timeout: int=10) -> Goals:
        """
        Start a proof session for a specific theorem in a Coq/Rocq file.
        """
        url = f"{self.base_url}/start"
        payload = {'session_id': self.session_id, 'file': file, 'thm': thm, 'failure': failure, 'timeout': timeout}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['state']
        else:
            raise ClientError(response.status_code, response.text)

    def get_session(self) -> Dict[str, Any]:
        url = f"{self.base_url}/get_session"
        payload = {'session_id': self.session_id}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output
        else:
            raise ClientError(response.status_code, response.text)