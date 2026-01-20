import os
import re
import subprocess
import time
import sys
from pathlib import Path
import psutil
from typing import Tuple, List

import pytest
import requests

from rocq_ml_toolbox.inference.client import PetClient
from rocq_ml_toolbox.parser.rocq_parser import RocqParser

def pytest_addoption(parser):
    parser.addoption(
        "--stress-workers",
        action="store",
        type=int,
        default=8,
        help="Number of client worker processes used inside the stress test."
    )
    parser.addoption(
        "--stress-n",
        action="store",
        type=int,
        default=10,
        help="Number of dataset entries to run in stress test (balanced success/non-success)."
    )
    parser.addoption(
        "--stdlib-path",
        type=str,
        default="/home/theo/.opam/mc_dev/lib/coq/user-contrib/Stdlib/",
        help="Location of the Stdlib in the current opam env."
    )


@pytest.fixture(scope="session")
def stress_workers(pytestconfig) -> int:
    return int(pytestconfig.getoption("--stress-workers"))

@pytest.fixture(scope="session")
def stress_n(pytestconfig) -> int:
    return int(pytestconfig.getoption("--stress-n"))

@pytest.fixture(scope="session")
def stdlib_path(pytestconfig) -> int:
    return str(pytestconfig.getoption("--stdlib-path"))

@pytest.fixture(scope="session")
def stdlib_filepaths(stdlib_path) -> List[str]:
    result = []
    for subdir, _, files in os.walk(stdlib_path):
        for file in files:
            if file.endswith('.v'):
                filepath = os.path.join(subdir, file)
                result.append(filepath)
    return result

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

def _start_redis():
    cmd = ["docker", "run", "-d", "-p", "6379:6379", "redis:latest"]
    container_id = subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()
    return container_id

def _stop_redis(container_id: str):
    subprocess.run(
        ["docker", "rm", "-f", container_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )

def _start_gunicorn(bind_host: str, bind_port: int) -> subprocess.Popen:
    kill_all_proc('gunicorn')
    kill_all_proc('pet-server')
    os.environ["NUM_PET_SERVER"] = str(4)
    os.environ["PET_SERVER_START_PORT"] = str(8765)
    os.environ["MAX_RAM_PER_PET"] = str(2048)
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"

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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
        text=True,
    )

@pytest.fixture(scope="session")
def host() -> str:
    host = "127.0.0.1"
    return host

@pytest.fixture(scope="session")
def port() -> int:
    port = 5100
    return port

@pytest.fixture(scope="session")
def server(host, port):
    # container_id = _start_redis()
    proc_gunicorn = _start_gunicorn(host, port)
    url = f"http://{host}:{port}"
    try:
        _wait_until_ready(url, proc=proc_gunicorn, timeout_s=30.0)
        yield "OK"
    finally:
        try:
            proc_gunicorn.terminate()
            proc_gunicorn.wait(timeout=2)
        except Exception:
            try:
                proc_gunicorn.kill()
                proc_gunicorn.wait(timeout=2)
            except Exception:
                pass
        
        # _stop_redis(container_id)

@pytest.fixture(scope="session")
def client(host, port, server) -> PetClient:
    client = PetClient(host, port)
    client.connect()
    return client

@pytest.fixture(scope="session")
def parser(client) -> RocqParser:
    parser = RocqParser(client)
    return parser