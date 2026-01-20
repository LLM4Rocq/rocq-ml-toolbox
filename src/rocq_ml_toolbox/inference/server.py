import os
import traceback
import logging

from flask import Flask, request, jsonify, current_app, send_file
from pytanque import PetanqueError
from pytanque.protocol import (
    Opts,
    Inspect
)

from ..rocq_lsp.client import LspClient
from ..rocq_lsp.structs import TextDocumentItem
from ..parser.utils.position import extract_subtext
from ..parser.ast.driver import generate_ast_dump_file
from .client import StateExtended
from .sessions import SessionManager, UnresponsiveError

app = Flask(__name__)

NUM_PET_SERVER = int(os.environ["NUM_PET_SERVER"])
PET_SERVER_START_PORT = int(os.environ["PET_SERVER_START_PORT"])
REDIS_URL = os.environ["REDIS_URL"]

session_manager = SessionManager(
    redis_url=REDIS_URL,
    pet_server_start_port=PET_SERVER_START_PORT,
    num_pet_server=NUM_PET_SERVER,
    timeout_ok=15
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
    session_id = session_manager.create_session()
    return jsonify({"session_id": session_id}), 200


@app.route("/get_state_at_pos", methods=["POST"])
def get_state_at_pos():
    """
    Get state at position.

    Expects JSON:
        - session_id (str)
        - filepath (str)
        - line (int)
        - character (int)
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "filepath", "line", "character"])
    if err is not None:
        return err

    if data['opts']:
        data['opts'] = Opts.from_json(data['opts'])
    state = session_manager.get_state_at_pos(**data)
    output = {"resp": state.to_json()}
    return jsonify(output), 200


@app.route("/run", methods=["POST"])
def run():
    """
    Execute a given tactic on the current proof state.

    Expects JSON:
        - state_ext: the current proof state (JSON from previous response)
        - tactic: tactic command (str)
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["state_ext", "tactic", "session_id"])
    if err is not None:
        return err
    
    data['state_ext'] = StateExtended.from_json(data['state_ext'])
    data['opts'] = Opts.from_json(data['opts']) if 'opts' in data and data['opts'] else None
    state = session_manager.run(**data)
    output = {"resp": state.to_json()}
    return jsonify(output), 200

@app.route("/goals", methods=["POST"])
def goals():
    """
    Gather goals associated to a state.

    Expects JSON:
        - state_ext: the current proof state (JSON from previous response)
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "state_ext"])
    if err is not None:
        return err
    
    data['state_ext'] = StateExtended.from_json(data['state_ext'])
    goals = session_manager.goals(**data)
    output = {"resp": [goal.to_json() for goal in goals]}
    return jsonify(output), 200

@app.route("/complete_goals", methods=["POST"])
def complete_goals():
    """
    Gather complete goals associated to a state.

    Expects JSON:
        - state_ext: the current proof state (JSON from previous response)
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "state_ext"])
    if err is not None:
        return err
    
    data['state_ext'] = StateExtended.from_json(data['state_ext'])
    goals = session_manager.complete_goals(**data)
    output = {"resp": goals.to_json()}
    return jsonify(output), 200

@app.route("/premises", methods=["POST"])
def premises():
    """
    Gather accessible premises (lemmas, definitions) from a state.

    Expects JSON:
        - state_ext: the current proof state (JSON from previous response)
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "state_ext"])
    if err is not None:
        return err
    
    data['state_ext'] = StateExtended.from_json(data['state_ext'])
    premises = session_manager.premises(**data)
    output = {"resp": premises}
    return jsonify(output), 200

@app.route("/state_equal", methods=["POST"])
def state_equal():
    """
    Check whether state st1 is equal to state st2.

    Expects JSON:
        - st1_ext: the first state
        - st2_ext: the second state
        - kind: comparison type
        - session_id: session ID from /login
    """
    
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "st1_ext", "st2_ext", "kind"])
    if err is not None:
        return err
    
    data['st1_ext'] = StateExtended.from_json(data['st1_ext'])
    data['st2_ext'] = StateExtended.from_json(data['st2_ext'])
    data['kind'] = Inspect.from_json(data['kind'])
    result = session_manager.state_equal(**data)
    output = {"resp": result}
    return jsonify(output), 200

@app.route("/state_hash", methods=["POST"])
def state_hash():
    """
    Get a hash value for a proof state.

    Expects JSON:
        - state_ext
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "state_ext"])
    if err is not None:
        return err
    
    data['state_ext'] = StateExtended.from_json(data['state_ext'])
    hash = session_manager.state_hash(**data)
    output = {"resp": hash}
    return jsonify(output), 200

@app.route("/toc", methods=["POST"])
def toc():
    """
    Get toc of a file.

    Expects JSON:
        - file
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "file"])
    if err is not None:
        return err
    
    toc = session_manager.toc(**data)
    output = {"resp": toc}
    return jsonify(output), 200

@app.route("/ast", methods=["POST"])
def ast():
    """
    Get ast of a command parsed at a state.

    Expects JSON:
        - text: command to parse
        - state_ext
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "text", "state_ext"])
    if err is not None:
        return err
    
    data['state_ext'] = StateExtended.from_json(data['state_ext'])
    ast = session_manager.ast(**data)
    output = {"resp": ast}
    return jsonify(output), 200

@app.route("/ast_at_pos", methods=["POST"])
def ast_at_pos():
    """
    Get ast at a specified position in a file.

    Expects JSON:
        - file
        - line
        - character
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "file", "line", "character"])
    if err is not None:
        return err
    
    ast = session_manager.ast_at_pos(**data)
    output = {"resp": ast}
    return jsonify(output), 200

@app.route("/get_root_state", methods=["POST"])
def get_root_state():
    """
    Get root state of a document.

    Expects JSON:
        - file
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "file"])
    if err is not None:
        return err
    
    data['opts'] = Opts.from_json(data['opts']) if 'opts' in data and data['opts'] else None
    state = session_manager.get_root_state(**data)
    output = {"resp": state.to_json()}
    return jsonify(output), 200

@app.route("/list_notations_in_statement", methods=["POST"])
def list_notations_in_statement():
    """
    Get the list of notations appearing in a theorem/lemma statement.

    Expects JSON:
        - state_ext
        - statement
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id", "state_ext", "statement"])
    if err is not None:
        return err
    
    data['state_ext'] = StateExtended.from_json(data['state_ext'])
    notations = session_manager.list_notations_in_statement(**data)
    output = {"resp": notations}
    return jsonify(output), 200

@app.route("/start", methods=["POST"])
def start():
    """
    Start a proof session for a specific theorem in a Coq/Rocq file.

    Expects JSON:
        - file
        - thm
        - session_id: session ID from /login
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["file", "thm"])
    if err is not None:
        return err
    
    data['opts'] = Opts.from_json(data['opts']) if 'opts' in data and data['opts'] else None
    state = session_manager.start(**data)
    output = {"resp": state.to_json()}
    return jsonify(output), 200

# @app.route("/query", methods=["POST"])
# def query():
#     """
#     Send a low-level JSON-RPC query to the server.

#     Expects JSON:
#         - params
#         - size
#         - session_id: session ID from /login
#     """
#     data = request.get_json(force=True, silent=False)
#     err = require_json_fields(data, ["params", "size"])
#     if err is not None:
#         return err
    
#     data['params'] = data['params'].to_json()
#     resp = session_manager.start(**data)
#     output = {"resp": resp.to_json()}
#     return jsonify(output), 200

@app.route("/get_document", methods=["POST"])
def get_document():
    """
    Extract fleche document representation from document at `path`.
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["path"])
    if err is not None:
        return err
    path = data["path"]
    with LspClient() as client:
        item = TextDocumentItem(path)
        client.initialize(item)
        client.didOpen(item)
        fleche_document = client.getDocument(item)
    
    text_utf8 = item.text.encode("utf-8")
    for ranged_span in fleche_document.spans:
        ranged_span.span = extract_subtext(text_utf8, ranged_span.range).decode("utf-8")
    return jsonify(fleche_document.to_json()), 200

@app.route("/get_ast", methods=["POST"])
def get_ast():
    """
    Extract full AST (verbatim) from document at `path`.
    """
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["path", "force_dump"])
    if err is not None:
        return err
    path = data["path"]
    dump_path = generate_ast_dump_file(path, force_dump=data['force_dump'])
    return send_file(dump_path, as_attachment=True)

@app.route("/get_session", methods=["POST"])
def get_session():
    data = request.get_json(force=True, silent=False)
    err = require_json_fields(data, ["session_id"])
    if err is not None:
        return err
    session_id = data["session_id"]
    return jsonify(session_manager.get_session(session_id).to_json()), 200