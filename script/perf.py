import os
import json
import concurrent.futures
import random
import subprocess
import time
from collections import defaultdict

from pytanque.client import PytanqueMode, Pytanque

NUM_ELEMENTS=360
NUM_STRESS_WORKERS=10
HOST="127.0.0.1"
PORT=8100
PET_SERVER_START_PORT = 8765

os.environ["NUM_PET_SERVER"] = str(NUM_STRESS_WORKERS)
os.environ["PET_SERVER_START_PORT"] = str(PET_SERVER_START_PORT)
os.environ["MAX_RAM_PER_PET"] = str(16_000)
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

def _start_arbiter() -> subprocess.Popen:
    cmd = [
        "python",
        "-m", "src.rocq_ml_toolbox.inference.arbiter"
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

def _start_uvicorn(bind_host: str, bind_port: int, workers:int) -> subprocess.Popen:
    cmd = [
        "uvicorn",
        "src.rocq_ml_toolbox.inference.server:app",
        # "--log-config", "src/rocq_ml_toolbox/inference/logging_config.yaml",
        "--host", bind_host,
        "--port", str(bind_port),
        "--workers", str(workers),
        "--timeout-worker-healthcheck", "600"
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

def try_proof(entry, host, port, mode: PytanqueMode) -> bool:
    client = Pytanque(host, port, mode=mode)
    client.connect()
    filepath = entry["filepath"]

    state = client.get_state_at_pos(filepath, entry["line"], entry["character"], timeout=60)
    for step in entry["steps"]:
        state = client.run(state, step, timeout=60)
    return True

def kill_proc(proc: subprocess.Popen):
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

with open('tests/pytest_toprove.json', 'r') as file:
    to_prove = json.load(file)

file_dict = defaultdict(list)

for entry in to_prove:
    file_dict[entry["filepath"]].append(entry)

subselection = []

for per_file_list in file_dict.values():
    subselection.append(random.choice(per_file_list))

selection = random.sample(subselection, k=NUM_ELEMENTS)

NUM_STRESS_WORKERS = 1
os.environ["NUM_PET_SERVER"] = str(1)
time.sleep(3)
proc_arbiter = _start_arbiter()
time.sleep(5)
start_t = time.time()
try:
    for entry in selection:
        try_proof(entry, HOST, PET_SERVER_START_PORT, PytanqueMode.SOCKET)
    delta_t = time.time() - start_t
    print(f"Total time for petanque: {delta_t}")
finally:
    kill_proc(proc_arbiter)

for num_stress in range(1, 16):
    NUM_STRESS_WORKERS = num_stress
    os.environ["NUM_PET_SERVER"] = str(NUM_STRESS_WORKERS)
    proc_arbiter = _start_arbiter()
    time.sleep(3)
    proc_uvicorn = _start_uvicorn(HOST, PORT, 2*NUM_STRESS_WORKERS+1)
    time.sleep(5)

    start_t = time.time()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_stress) as ex:
            future_to_expected = [
                ex.submit(try_proof, entry, HOST, PORT, PytanqueMode.HTTP) for entry in selection
            ]

        for future in concurrent.futures.as_completed(future_to_expected):
            assert future.result()
        
        delta_t = time.time() - start_t
        print(f"Total time for rocq-ml-server ({NUM_STRESS_WORKERS} threads): {delta_t}")
    finally:
        kill_proc(proc_arbiter)
        kill_proc(proc_uvicorn)
    time.sleep(3)