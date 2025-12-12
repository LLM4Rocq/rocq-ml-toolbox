import os
import json
import concurrent.futures
from pathlib import Path

import redis
import pytest
from filelock import FileLock

from inference_server.client import PetClient
from inference_server.redis_keys import pet_status_key, PetStatus


MC_DIR = os.environ.get("MC_DIR", "stress_test_light/source")
redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def kill_signal(pet_idx):
    redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)

def try_proof_kill(entry, server_url):
    client = PetClient(server_url)
    filepath = os.path.join(MC_DIR, entry['filepath'])
    try:
        state, _ = client.start_thm(filepath, entry['line'], entry['character'])
        for step in entry['proof_steps']:
            sess = client.get_session()
            kill_signal(sess['pet_idx'])
            state, _ = client.run(state, step)
    except Exception as e:
        return False
    
    return True

def _cache_paths(n: int) -> tuple[Path, FileLock]:
    cache = Path(__file__).with_name(f".crrracq_subset_balance_n{n}.json")
    lock = FileLock(str(cache) + ".lock")
    return cache, lock

def _cache_paths_valid(n: int) -> tuple[Path, FileLock]:
    cache = Path(__file__).with_name(f".crrracq_subset_valid_n{n}.json")
    lock = FileLock(str(cache) + ".lock")
    return cache, lock

def _load_subset_balanced(n: int):
    """
    Select n entries, balanced between SUCCESS and non-SUCCESS, with early stopping.
    Cached to disk to avoid rescanning the dataset on every pytest run.
    """
    cache, lock = _cache_paths(n)

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
    cache, lock = _cache_paths_valid(n)

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

def try_proof(entry, server_url: str) -> bool:
    """
    Top-level function so it can be pickled for ProcessPoolExecutor.
    Mirrors your CLI script behavior.
    """
    client = PetClient(server_url)
    filepath = os.path.join(MC_DIR, entry["filepath"])

    k = 0
    try:
        state, _ = client.start_thm(filepath, entry["line"], entry["character"])
        for k, step in enumerate(entry["proof_steps"], start=1):
            state, _ = client.run(state, step)
    except Exception:
        if k != len(entry["proof_steps"]) or entry["status"] == "SUCCESS":
            return False
        return True

    return entry["status"] == "SUCCESS"


@pytest.mark.stress
def test_stress_dataset(server_url, stress_workers, stress_n):
    selection = _load_subset_balanced(stress_n)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=stress_workers) as ex:
        futures = [ex.submit(try_proof, entry, server_url) for entry in selection]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    assert all(results), f"Some proofs failed: {results.count(False)} / {len(results)}"

@pytest.mark.replay
def test_replay_dataset(server_url, stress_n):
    selection = _load_subset_valid(stress_n)

    results = []
    for entry in selection:
        result = try_proof(entry, server_url)
        results.append(result)

    assert all(results), f"Some proofs failed: {results.count(False)} / {len(results)}"
