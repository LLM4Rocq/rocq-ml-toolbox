import os
import re
import subprocess
import time
import sys
from pathlib import Path
import psutil

import pytest
import requests

from src.rocq_ml_toolbox.inference.client import ClientError, PetClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--server-url",
        action="store",
        default=None,
        help="Use an already running server, e.g. http://127.0.0.1:5000",
    )
    parser.addoption(
        "--stress-workers",
        action="store",
        type=int,
        default=8,
        help="Number of client worker processes used inside the stress test.",
    )
    parser.addoption(
        "--stress-n",
        action="store",
        type=int,
        default=100,
        help="Number of dataset entries to run in stress test (balanced success/non-success).",
    )


@pytest.fixture(scope="session")
def stress_workers(pytestconfig) -> int:
    return int(pytestconfig.getoption("--stress-workers"))


@pytest.fixture(scope="session")
def stress_n(pytestconfig) -> int:
    return int(pytestconfig.getoption("--stress-n"))


def _worker_index(request) -> int:
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    m = re.match(r"gw(\d+)$", worker_id)
    return int(m.group(1)) if m else 0


def _wait_until_ready(url: str, proc: subprocess.Popen | None, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None

    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("Gunicorn exited early while starting.")

        try:
            r = requests.get(url + "/health", timeout=0.5)
            if r.status_code == 200 and r.text.strip() == "OK":
                return
        except Exception as e:
            last_err = e

        time.sleep(0.2)

    raise TimeoutError(f"Server not ready at {url}. Last error: {last_err!r}")

def kill_all_proc(proc_name):
    for proc in psutil.process_iter():
        if proc.name() == proc_name:
            proc.kill()
    time.sleep(1)

def _start_gunicorn(bind_host: str, bind_port: int) -> subprocess.Popen:
    kill_all_proc('gunicorn')
    kill_all_proc('pet-server')
    cmd = [
        "gunicorn",
        "-w", "9",
        "-b", f"{bind_host}:{bind_port}",
        "src.rocq_ml_toolbox.inference.server:app",
        "-t", "600",
        "-c", "python:src.rocq_ml_toolbox.inference.gunicorn_config",
        "--error-logfile", "gunicorn-error.log",
        "--access-logfile", "gunicorn-access.log",
        "--capture-output",
    ]
    env = os.environ.copy()
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,   # avoid pytest capture overhead
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        text=True,
    )

@pytest.fixture(scope="session")
def server_url(pytestconfig, request) -> str:
    provided = pytestconfig.getoption("--server-url")
    if provided:
        url = provided.rstrip("/")
        _wait_until_ready(url, proc=None, timeout_s=30.0)
        yield url
        return

    # Auto-start (fallback): one server per xdist worker, on different ports
    widx = _worker_index(request)
    host = "127.0.0.1"
    port = 5100 + widx

    proc = _start_gunicorn(host, port)
    url = f"http://{host}:{port}"

    try:
        _wait_until_ready(url, proc=proc, timeout_s=30.0)
        yield url
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass

@pytest.fixture(scope="session")
def client(server_url):
    client = PetClient(server_url)
    client.connect()
    return client
