import json
import concurrent.futures
import argparse
import os

from datasets import load_dataset
import redis
from tqdm import tqdm

from src.rocq_ml_toolbox.inference.client import PetClient


redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
MC_DIR = 'stress_test_light/source'

def try_proof(entry, server_url):
    client = PetClient(server_url)
    client.connect()
    filepath = os.path.join(MC_DIR, entry['filepath'])
    try:
        k = 0
        state, _ = client.start_thm(filepath, entry['line'], entry['character'])
        for k, step in enumerate(entry['proof_steps'], start=1):
            state, _ = client.run(state, step)
    except Exception as e:
        if k != len(entry['proof_steps']) or entry['status'] == 'SUCCESS':
            return False
        else:
            return True
    return entry['status'] == 'SUCCESS'

parser = argparse.ArgumentParser(description="Stress test-CLI")
parser.add_argument(
    "--server-url",
    type=str,
    default="http://127.0.0.1:5000"
)
parser.add_argument(
    "--max-workers",
    type=int,
    default=1
)
args = parser.parse_args()

ds = load_dataset('theostos/crrracq')

selection_valid = []
selection_incorrect = []

for entry in ds['train']:
    if entry['status'] == 'SUCCESS':
        selection_valid.append(entry)

for entry in ds['train']:
    if entry['status'] != 'SUCCESS':
        selection_incorrect.append(entry)
    if len(selection_incorrect) == len(selection_valid):
        break

selection = selection_incorrect + selection_valid

futures = []
true_count = 0
false_count = 0

with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    futures = [executor.submit(try_proof, entry, args.server_url) for entry in selection]

    for f in tqdm(concurrent.futures.as_completed(futures),
                  position=1, total=len(futures)):
        try:
            result = f.result()
        except Exception:
            result = False

        if result:
            true_count += 1
        else:
            false_count += 1

print("True:", true_count)
print("False:", false_count)