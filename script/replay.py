import json
import concurrent.futures
import argparse
import os
import random
import time
import psutil

from datasets import load_dataset

import redis
from tqdm import tqdm

from src.rocq_ml_toolbox.inference.client import PetClient
from src.rocq_ml_toolbox.inference.redis_keys import pet_status_key, PetStatus

redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

MC_DIR = 'stress_test_light/source'

def kill_all_pet(proc_name='pet-server'):
    for proc in psutil.process_iter():
        if proc.name() == proc_name:
            proc.kill()
    time.sleep(1)

def kill_signal(pet_idx):
    redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)

def try_proof(entry, server_url, retry=1, failure_rate=0.3):
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
        state = call_with_failover(
            client.get_state_at_pos,
            filepath,
            entry["line"],
            entry["character"],
            timeout=120
        )

        for step in entry["proof_steps"]:

            state = call_with_failover(
                client.run,
                state,
                step,
                timeout=60
            )
        return True

    except Exception as e:
        print(e)
        print("ISSUE")
        return False

parser = argparse.ArgumentParser(description="Stress test-CLI")
parser.add_argument(
    "--server-url",
    type=str,
    default="http://127.0.0.1:5000"
)
parser.add_argument(
    "--max-workers",
    type=int,
    default=8
)
args = parser.parse_args()

ds = load_dataset('theostos/crrracq')

selection_valid = []
random.shuffle(ds)
for entry in ds['train']:
    if entry['status'] == 'SUCCESS':
        selection_valid.append(entry)

results = []

with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    futures = [executor.submit(try_proof, entry, args.server_url) for entry in selection_valid]

    for f in tqdm(concurrent.futures.as_completed(futures),
                  position=1, total=len(futures)):
        try:
            result = f.result()
        except Exception:
            result = False
        
        results.append(result)


print(f"Issues: {results.count(False)} / {len(results)}")