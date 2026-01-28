import os
import json
import concurrent.futures
from pathlib import Path
from typing import Optional, List
import random
import psutil
import time
import threading

from redis import Redis
import pytest
from filelock import FileLock

from rocq_ml_toolbox.parser.rocq_parser import Source, Theorem, VernacElement, RocqParser
from pytanque import Pytanque
from pytanque.client import PytanqueMode
from rocq_ml_toolbox.inference.redis_keys import pet_status_key, PetStatus
from rocq_ml_toolbox.parser.utils.position import offset_to_pos


def kill_all_proc(proc_name: str):
    for proc in psutil.process_iter():
        if proc.name() == proc_name:
            proc.kill()
    time.sleep(1)

def wait_pause(redis_client: Redis):
    if redis_client.get("ctrl:state") != "pause":
        return

    redis_client.incr("ctrl:paused_workers")
    try:
        while redis_client.get("ctrl:state") == "pause":
            time.sleep(0.2)
    finally:
        redis_client.decr("ctrl:paused_workers")

def try_proof_control(entry, host, port, redis_url: str) -> bool:
    redis_client = Redis.from_url(redis_url, decode_responses=True)
    redis_client.incr("ctrl:num_workers")
    client = Pytanque(host, port, mode=PytanqueMode.HTTP)
    client.connect()
    wait_pause(redis_client)
    state = client.get_state_at_pos(entry['filepath'], entry['line'], entry['character'], timeout=60)
    try:
        for step in entry['steps']:
            wait_pause(redis_client)
            state = client.run(state, step, timeout=60)
            redis_client.incr("ctrl:tasks_done")
    finally:
        redis_client.decr("ctrl:num_workers")

    return True

def coordinator(redis_url: str, fail_each_k=100, stop_event: threading.Event | None = None):
    redis_client = Redis.from_url(redis_url, decode_responses=True)

    while stop_event is None or not stop_event.is_set():
        time.sleep(0.1)

        tasks_done = int(redis_client.get("ctrl:tasks_done") or 0)
        if tasks_done <= fail_each_k:
            continue

        redis_client.set("ctrl:state", "pause")

        once = False
        while stop_event is None or not stop_event.is_set():
            num_workers = int(redis_client.get("ctrl:num_workers") or 0)
            paused_workers = int(redis_client.get("ctrl:paused_workers") or 0)

            if num_workers == paused_workers:
                if once:
                    time.sleep(5)
                else:
                    break

            time.sleep(0.02)  # prevent busy spin

        kill_all_proc("pet-server")
        time.sleep(5)
        redis_client.set("ctrl:tasks_done", 0)
        redis_client.set("ctrl:state", "resume")
    
def try_proof(entry, host, port) -> bool:
    client = Pytanque(host, port, mode=PytanqueMode.HTTP)
    client.connect()
    filepath = entry["filepath"]

    state = client.get_state_at_pos(filepath, entry["line"], entry["character"])
    for step in entry["steps"]:
        state = client.run(state, step, timeout=15)
    return True

@pytest.fixture(scope="session")
def to_prove(parser: RocqParser, stdlib_filepaths: List[str]):
    cache_file = 'tests/pytest_toprove.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as file:
            return json.load(file)
    
    result = []
    for filepath in stdlib_filepaths:
        source = Source.from_local_path(filepath)
        for element, steps in parser.extract_proofs_wo_check(source):
            pos = offset_to_pos(source.content_utf8, element.span.ep)
            entry = {
                "filepath": source.path,
                "line": pos.line,
                "character": pos.character,
                "steps": steps
            }
            result.append(entry)
    
    with open(cache_file, 'w') as file:
        json.dump(result, file, indent=4)
    return result

@pytest.mark.validation
def test_validation(host, port, stress_workers, to_prove, stress_n):
    selection = random.sample(to_prove, k=stress_n)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=stress_workers) as ex:
        future_to_expected = [
            ex.submit(try_proof, entry, host, port) for entry in selection
        ]

    for future in concurrent.futures.as_completed(future_to_expected):
        assert future.result()

@pytest.mark.replay_mono
def test_replay_mono(host, port, stress_workers, to_prove, stress_n):
    selection = random.sample(to_prove, k=stress_n)
    
    for entry in selection:
        client = Pytanque(host, port, mode=PytanqueMode.HTTP)
        client.connect()
        filepath = entry["filepath"]

        state = client.get_state_at_pos(filepath, entry["line"], entry["character"])
        if random.random() < 0.3:
            kill_all_proc("pet-server")
            time.sleep(5)
        for step in entry["steps"]:
            state = client.run(state, step, timeout=15)
            if random.random() < 0.3:
                kill_all_proc("pet-server")
                time.sleep(5)

@pytest.mark.replay
def test_replay(host, port, stress_workers, to_prove, stress_n, redis_url):
    selection = random.sample(to_prove, k=stress_n)

    # Reset control keys for a clean run
    rc = Redis.from_url(redis_url, decode_responses=True)
    rc.set("ctrl:state", "resume")
    rc.set("ctrl:tasks_done", 0)
    rc.set("ctrl:num_workers", 0)
    rc.set("ctrl:paused_workers", 0)

    stop_event = threading.Event()
    coord_thread = threading.Thread(
        target=coordinator,
        args=(redis_url,),
        kwargs={"fail_each_k": 100, "stop_event": stop_event},
        daemon=True,
    )
    coord_thread.start()

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=stress_workers) as ex:
            futures = [
                ex.submit(try_proof_control, entry, host, port, redis_url)
                for entry in selection
            ]
            for f in concurrent.futures.as_completed(futures):
                assert f.result()
    finally:
        stop_event.set()
        coord_thread.join(timeout=1.0)
