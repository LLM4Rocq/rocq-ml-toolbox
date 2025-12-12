from typing import Dict, Tuple, List, Any

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
    
    def start_thm(self, filepath: str, line: int, character: int) -> Tuple[State, Goals]:
        """
        Start a theorem proving session for the theorem at the given index.
        """
        url = f"{self.base_url}/start_thm"
        payload = {'session_id': self.session_id, 'filepath': filepath, 'line': line, 'character': character}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['state'], output['goals']
        else:
            raise ClientError(response.status_code, response.text)

    def run(self, state: State, tactic: str) -> Tuple[State, Goals]:
        """
        Execute a given tactic on the current proof state.
        """
        url = f"{self.base_url}/run"
        payload = {'session_id': self.session_id, 'state': state, 'tactic': tactic}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()
            return output['state'], output['goals']
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