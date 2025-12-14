import os
import traceback
import logging

from flask import Flask, request, jsonify, current_app
from pytanque import State, PetanqueError

from .sessions import SessionManager, UnresponsiveError

app = Flask(__name__)

NUM_PET_SERVER = int(os.environ["NUM_PET_SERVER"])
PET_SERVER_START_PORT = int(os.environ["PET_SERVER_START_PORT"])
REDIS_URL = os.environ["REDIS_URL"]

session_manager = SessionManager(
    redis_url=REDIS_URL,
    pet_server_start_port=PET_SERVER_START_PORT,
    num_pet_server=NUM_PET_SERVER,
    timeout_start_thm=120,
    timeout_run=60,
)

gunicorn_error_logger = logging.getLogger("gunicorn.error")
if gunicorn_error_logger.handlers:
    app.logger.handlers = gunicorn_error_logger.handlers
    app.logger.setLevel(gunicorn_error_logger.level)
    app.logger.propagate = False

    logging.getLogger("werkzeug").handlers = gunicorn_error_logger.handlers
    logging.getLogger("werkzeug").setLevel(gunicorn_error_logger.level)

def _traceback_str(e: Exception) -> str | None:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

def _json_error(error_code: str, message: str, status: int, exc: Exception | None = None):
    payload = {"error": error_code, "message": message}
    if exc is not None:
        tb = _traceback_str(exc)
        if tb is not None:
            payload["traceback"] = tb
    return jsonify(payload), status

# -------------------------------------------------------------------------
# Health
# -------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    if session_manager.pet_status():
        return "OK", 200
    return "Pet servers not ready", 503

# -------------------------------------------------------------------------
# Error handlers
# -------------------------------------------------------------------------

@app.errorhandler(PetanqueError)
def handle_petanque_error(e: PetanqueError):
    current_app.logger.error("PetanqueError", exc_info=e)
    return _json_error("petanque_error", str(e), 400, e)

@app.errorhandler(UnresponsiveError)
def handle_unresponsive_error(e: UnresponsiveError):
    current_app.logger.error("UnresponsiveError", exc_info=e)
    return _json_error("unresponsive", str(e), 503, e)

@app.errorhandler(KeyError)
def handle_key_error(e: KeyError):
    current_app.logger.error("KeyError", exc_info=e)
    return _json_error("not_found", str(e), 404, e)

@app.errorhandler(Exception)
def handle_unexpected_error(e: Exception):
    current_app.logger.error("Unhandled exception in request", exc_info=e)
    return _json_error("internal_error", f"internal server error: {e}", 500, e)


# -------------------------------------------------------------------------
# Utility: input validation
# -------------------------------------------------------------------------

def require_json_fields(data, required):
    missing = [k for k in required if k not in data]
    if missing:
        return (
            jsonify(
                {
                    "error": "bad_request",
                    "message": "missing required fields",
                    "missing": missing,
                }
            ),
            400,
        )
    return None

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.route("/login", methods=["GET"])
def login():
    """
    Return a session object with assigned pet-server index and unique session ID.
    """
    sess = session_manager.create_session()
    return jsonify({"session_id": sess.session_id}), 200


@app.route("/start_thm", methods=["POST"])
def start_thm():
    """
    Start a theorem.

    Expects JSON:
        - session_id (str)
        - filepath (str)
        - line (int)
        - character (int)
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "filepath", "line", "character", "failure"])
    if err is not None:
        return err

    state, goals = session_manager.start_thm(
        session_id=data["session_id"],
        filepath=data["filepath"],
        line=data["line"],
        character=data["character"],
        failure=data['failure']
    )
    goals_json = [goal.to_json() for goal in goals]
    output = {"state": state.to_json(), "goals": goals_json}
    return jsonify(output), 200


@app.route("/run", methods=["POST"])
def run():
    """
    Execute a given tactic on the current proof state.

    Expects JSON:
        - state: the current proof state (JSON from previous response)
        - tactic: tactic command (str)
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["state", "tactic", "session_id", "failure"])
    if err is not None:
        return err
        
    state, goals = session_manager.run(
        session_id=data["session_id"],
        state=State.from_json(data["state"]),
        tactic=data["tactic"],
        failure=data["failure"]
    )
    goals_json = [goal.to_json() for goal in goals]
    output = {"state": state.to_json(), "goals": goals_json}
    return jsonify(output), 200

@app.route("/get_session", methods=["POST"])
def get_session():
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id"])
    if err is not None:
        return err
    session_id = data["session_id"]
    return jsonify(session_manager.get_session(session_id).to_dict()), 200