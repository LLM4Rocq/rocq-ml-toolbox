import json
import os
import time
import threading

from pytanque import State
from flask import Flask, request, jsonify

from .sessions import SessionManager

app = Flask(__name__)

session_manager = SessionManager(
    pet_server_start_port=8765,
    num_pet_server=8,
    timeout_start_thm=30,
    timeout_run=10
)

@app.route('/health', methods=['GET'])
def health():
    if session_manager.pet_status():
        return "OK", 200
    else:
        return "Pet servers not ready", 500

@app.route('/login', methods=['GET'])
def login():
    """
    Return a session object with assigned pet-server index and unique session ID.

    Returns:
            - status_code
            - output: the assigned session ID
    """
    try:
        sess = session_manager.create_session()
        return jsonify({"session_id": sess.session_id}), 200
    except Exception as e:
        return str(e), 500

@app.route('/start_thm', methods=['POST'])
def start_thm():
    """
    Start a theorem

    Expects:
        - session_id (str): the session ID assigned from /login.
        - thm_name (str): the name of the theorem to start.
        - filepath (str): the file path where the theorem is located.
        - line (int): the line number of the theorem.
        - character (int): the character position of the theorem.

    Returns:
            - status_code
            - output: A dictionary containing:
                - state: The initial proof state (in JSON format)
                - goals: A list of pretty-printed goals
    """
    try:
        data = request.get_json()
        state, goals = session_manager.start_thm(**data)
        goals_json = [goal.to_json() for goal in goals]
        output = {"state": state.to_json(), "goals": goals_json}
        return jsonify(output), 200
    except Exception as e:
        return str(e), 500

@app.route('/run', methods=['POST'])
def run():
    """
    Execute a given tactic on the current proof state.

    Expects:
        - state: the current proof state.
        - tactic: the tactic command to execute.
        - session_id: the session ID assigned from /login.

    Returns:
            - status_code
            - output:
                - state: new proof state
                - goals: goals
    """
    try:
        data = request.get_json()        
        current_state = State.from_json(data['state'])
        tactic = data['tactic']
        session_id = data['session_id']
        state, goals = session_manager.run(session_id, current_state, tactic)
        goals_json = [goal.to_json() for goal in goals]
        output = {"state": state.to_json(), "goals": goals_json}
        return jsonify(output), 200
    except Exception as e:
        return str(e), 500
