import os
import json
import concurrent.futures
from pathlib import Path
from typing import Optional
import random
import psutil
import time

import redis
import pytest
from filelock import FileLock

from src.rocq_ml_toolbox.inference.client import PetClient
from src.rocq_ml_toolbox.inference.redis_keys import pet_status_key, PetStatus


MC_DIR = os.environ.get("MC_DIR", "stress_test_light/source")
redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

def kill_all_proc(proc_name: str):
    for proc in psutil.process_iter():
        if proc.name() == proc_name:
            proc.kill()
    time.sleep(1)

def try_proof_kill(entry, server_url: str) -> bool:
    client = PetClient(server_url)
    filepath = os.path.join(MC_DIR, entry['filepath'])
    try:
        state, _ = client.start_thm(filepath, entry['line'], entry['character'])
        for step in entry['proof_steps']:
            kill_all_proc('pet-server')
            state, _ = client.run(state, step)
    except Exception as e:
        return False
    
    return True

def _cache_paths(n: int) -> tuple[Path, FileLock]:
    cache = Path(__file__).with_name(f".crrracq_subset_balance_n{n}.json")
    lock = FileLock(str(cache) + ".lock")
    return cache, lock, '.crrracq_full.json'

def _cache_paths_valid(n: int) -> tuple[Path, FileLock]:
    cache = Path(__file__).with_name(f".crrracq_subset_valid_n{n}.json")
    lock = FileLock(str(cache) + ".lock")
    return cache, lock, '.crrracq_full.json'

def _load_subset_balanced(n: int):
    """
    Select n entries, balanced between SUCCESS and non-SUCCESS, with early stopping.
    Cached to disk to avoid rescanning the dataset on every pytest run.
    """
    cache, lock, full_ds = _cache_paths(n)

    with lock:
        if cache.exists():
            return json.loads(cache.read_text())

        from datasets import load_dataset
        ds = load_dataset("theostos/crrracq", split="train")

        need_ok = n // 2
        need_bad = n - need_ok
        ok, bad = [], []

        for x in ds:
            if x["status"] == "SUCCESS":
                if len(ok) < need_ok:
                    ok.append(x)
            else:
                if len(bad) < need_bad:
                    bad.append(x)

            if len(ok) == need_ok and len(bad) == need_bad:
                break

        subset = ok + bad

        payload = [
            {
                "filepath": x["filepath"],
                "line": x["line"],
                "character": x["character"],
                "status": x["status"],
                "proof_steps": x["proof_steps"],
            }
            for x in subset
        ]
        cache.write_text(json.dumps(payload))
        return payload

def _load_subset_valid(n: int):
    """
    Select n entries.
    Cached to disk to avoid rescanning the dataset on every pytest run.
    """
    cache, lock, full_ds = _cache_paths_valid(n)

    with lock:
        if cache.exists():
            return json.loads(cache.read_text())

        from datasets import load_dataset
        ds = load_dataset("theostos/crrracq", split="train")

        ok = []

        for x in ds:
            if x["status"] == "SUCCESS":
                if len(ok) < n:
                    ok.append(x)
            if len(ok) == n:
                break


        payload = [
            {
                "filepath": x["filepath"],
                "line": x["line"],
                "character": x["character"],
                "status": x["status"],
                "proof_steps": x["proof_steps"],
            }
            for x in ok
        ]
        cache.write_text(json.dumps(payload))
        return payload

def try_proof(entry, server_url, retry=1, failure_rate=0.) -> bool:
    client = PetClient(server_url)
    filepath = os.path.join(MC_DIR, entry["filepath"])

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

    try:
        state, _ = call_with_failover(
            client.start_thm,
            filepath,
            entry["line"],
            entry["character"],
        )

        for step in entry["proof_steps"]:

            state, _ = call_with_failover(
                client.run,
                state,
                step
            )
        return True

    except Exception as e:
        return False


@pytest.mark.validation
def test_validation(server_url, stress_workers, stress_n):
    selection = _load_subset_balanced(stress_n)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=stress_workers) as ex:
        futures = [ex.submit(try_proof, entry, server_url) for entry in selection]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert all(results), f"Some proofs failed: {results.count(False)} / {len(results)}"

@pytest.mark.replay
def test_replay(server_url, stress_workers, stress_n):
    selection = _load_subset_valid(stress_n)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=stress_workers) as ex:
        futures = [ex.submit(try_proof, entry, server_url, failure_rate=0.3) for entry in selection]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert all(results), f"Some proofs failed: {results.count(False)} / {len(results)}"

@pytest.mark.manual_kill
def test_manual_kill(server_url, stress_n):
    selection = _load_subset_valid(stress_n)

    results = []
    for entry in selection:
        results.append(try_proof_kill(entry, server_url))

    assert all(results), f"Some proofs failed: {results.count(False)} / {len(results)}"
