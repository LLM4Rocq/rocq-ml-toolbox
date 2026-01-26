import os
import json
import concurrent.futures
from pathlib import Path
from typing import Optional, List
import random
import psutil
import time

import redis
import pytest
from filelock import FileLock

from rocq_ml_toolbox.parser.rocq_parser import Source, Theorem, VernacElement, RocqParser
from pytanque import Pytanque as PetClient
from pytanque.client import PytanqueMode
from rocq_ml_toolbox.inference.redis_keys import pet_status_key, PetStatus
from rocq_ml_toolbox.parser.utils.position import offset_to_pos


def kill_all_proc(proc_name: str):
    for proc in psutil.process_iter():
        if proc.name() == proc_name:
            proc.kill()
    time.sleep(1)

def try_proof_kill(entry, host, port) -> bool:
    client = PetClient(host, port, mode=PytanqueMode.HTTP)
    client.connect()
    state = client.get_state_at_pos(entry['filepath'], entry['line'], entry['character'])
    for step in entry['steps']:
        kill_all_proc('pet-server')
        state = client.run(state, step)
    return True

def try_proof(entry, host, port, retry=1, failure_rate=0.) -> bool:
    client = PetClient(host, port)
    client.connect()
    filepath = entry["filepath"]

    retry = max(1, int(retry))

    def call_with_failover(fn, *args, **kwargs):
        """
        Keep trying until:
          - the call succeeds, OR
          - we hit `retry` failures where failure was NOT simulated.
        Simulated failures do not stop the loop.
        """
        real_failures = 0
        failure = (random.random() < failure_rate)
        while True:
            try:
                return fn(*args, failure=failure, **kwargs)
            except Exception as e:
                if failure:
                    return fn(*args, failure=False, **kwargs)
                real_failures += 1
                if real_failures >= retry:
                    raise  # bubble up to return False

    state = call_with_failover(
        client.get_state_at_pos,
        filepath,
        entry["line"],
        entry["character"],
    )

    for step in entry["steps"]:

        state = call_with_failover(
            client.run,
            state,
            step
        )
    return True

@pytest.fixture(scope="session")
def to_prove(parser: RocqParser, stdlib_filepaths: List[str]):
    result = []
    for filepath in stdlib_filepaths:
        source = Source.from_local_path(filepath)
        for element, steps in parser.extract_proofs_wo_check(source):
            pos = offset_to_pos(source.content_utf8, element.span.ep)
            entry = {
                "filepath": source.path,
                "line": pos.line,
                "character": pos.character,
                "c_line": pos.line,
                "steps": steps
            }
            result.append(entry)
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

@pytest.mark.replay
def test_replay(host, port, stress_workers, to_prove, stress_n):
    selection = random.sample(to_prove, k=stress_n)

    with concurrent.futures.ProcessPoolExecutor(max_workers=stress_workers) as ex:
        futures = [ex.submit(try_proof, entry, host, port, failure_rate=0.3) for entry in selection]
    for f in concurrent.futures.as_completed(futures):
        assert f.result()


@pytest.mark.manual_kill
def test_manual_kill(host, port, to_prove, stress_n):
    selection = random.sample(to_prove, k=stress_n)

    for entry in selection:
        assert try_proof_kill(entry, host, port)
