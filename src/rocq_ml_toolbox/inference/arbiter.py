from __future__ import annotations

import os
import json
import time
import signal
import threading
import subprocess
from typing import List, Optional

import psutil
import redis

from .redis_keys import (
    monitor_epoch_key,
    pet_status_key,
    generation_key,
    pet_lock_key,
    PetStatus,
    ALL_KEYS_STAR,
)

NUM_PET_SERVER = int(os.environ["NUM_PET_SERVER"])
PET_SERVER_START_PORT = int(os.environ["PET_SERVER_START_PORT"])
MAX_RAM_PER_PET = int(os.environ["MAX_RAM_PER_PET"])  # MB; 0 disables RAM checks
REDIS_URL = os.environ["REDIS_URL"]

redis_client = redis.Redis.from_url(REDIS_URL)

pet_servers: List[Optional[subprocess.Popen]] = [None] * NUM_PET_SERVER
monitor_threads: list[threading.Thread] = []

_stop_event = threading.Event()


def clean_redis_all() -> None:
    for key in ALL_KEYS_STAR:
        for subkey in redis_client.scan_iter(key):
            redis_client.delete(subkey)


def kill_all_pet(proc_name: str = "pet-server") -> None:
    """Kill all existing pet-server processes (safety on startup)."""
    for proc in psutil.process_iter():
        try:
            if proc.name() == proc_name:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(1)


def start_pet_servers() -> None:
    """Spawn one pet-server process per pet_idx on fixed ports."""
    global pet_servers

    for pet_idx in range(NUM_PET_SERVER):
        port = PET_SERVER_START_PORT + pet_idx
        p = subprocess.Popen(["pet-server", "-p", str(port)])
        pet_servers[pet_idx] = p

    time.sleep(3)
    for pet_idx in range(NUM_PET_SERVER):
        redis_client.set(pet_status_key(pet_idx), PetStatus.OK)
        redis_client.set(generation_key(pet_idx), 0)

    print(
        "[arbiter] Started pet-servers:",
        [(pet_idx, PET_SERVER_START_PORT + pet_idx, p.pid) for pet_idx, p in enumerate(pet_servers) if p],
        flush=True,
    )


def stop_pet_servers() -> None:
    """Terminate all currently tracked pet-servers."""
    global pet_servers
    for pet_idx, p in enumerate(pet_servers):
        if p is None:
            continue
        try:
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
                p.wait(timeout=2)
            except Exception:
                pass
        pet_servers[pet_idx] = None
        redis_client.set(pet_status_key(pet_idx), "DOWN")

    print("[arbiter] Stopped all pet-servers", flush=True)


def restart_single_pet_server(pet_idx: int) -> None:
    """Restart the pet-server for a single index."""
    global pet_servers

    # Stop old process if present
    p = pet_servers[pet_idx]
    if p is not None:
        try:
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
                p.wait(timeout=2)
            except Exception:
                pass

    port = PET_SERVER_START_PORT + pet_idx
    new_p = subprocess.Popen(["pet-server", "-p", str(port)])
    pet_servers[pet_idx] = new_p

    gen_raw = redis_client.get(generation_key(pet_idx))
    generation = int(gen_raw) if gen_raw is not None else 0
    redis_client.set(generation_key(pet_idx), generation + 1)

    time.sleep(3.0)
    redis_client.set(pet_status_key(pet_idx), PetStatus.OK)
    print(f"[arbiter] Restarted pet-server idx={pet_idx} on port {port}, pid={new_p.pid}", flush=True)

def monitor_ram(poll_interval=0.1) -> None:
    """
    Monitor RAM usage
    """
    print(f"[arbiter] RAM Monitor started", flush=True)

    while not _stop_event.is_set():
        try:
            for pet_idx in range(NUM_PET_SERVER):
                p = pet_servers[pet_idx]
                if MAX_RAM_PER_PET > 0:
                    try:
                        proc = psutil.Process(p.pid)
                        rss_mb = proc.memory_info().rss / (1024 * 1024)
                        if rss_mb > MAX_RAM_PER_PET:
                            print(
                                f"[arbiter] pet-server idx={pet_idx} over RAM limit: "
                                f"{rss_mb:.1f} MB > {MAX_RAM_PER_PET} MB; scheduling restart",
                                flush=True,
                            )
                            redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(f"[arbiter] RAM check failed for pet_idx={pet_idx}: {e}", flush=True)
                
                time.sleep(poll_interval)
        except Exception as e:
            print(f"[arbiter] Error in RAM monitor: {e}", flush=True)
            time.sleep(poll_interval)

    print(f"[arbiter] Monitor thread exiting for pet_idx={pet_idx}", flush=True)
def monitor_redis_for_restarts(pet_idx: int) -> None:
    """
    Monitor Redis pet_status:{idx} keys for RESTART_NEEDED, detect crashes,
    and restart the corresponding pet-server.
    """
    print(f"[arbiter] Monitor thread started for pet_idx={pet_idx}", flush=True)
    ps = redis_client.pubsub()
    ps.subscribe(f"arbiter:req:{pet_idx}")

    while not _stop_event.is_set():
        try:
            for msg in ps.listen():
                if msg["type"] != "message":
                    continue
            
                req = json.loads(msg["data"])
                reply_channel = req["reply_to"]
                req_id = req["id"]

                p = pet_servers[pet_idx]
                # 1) Detect crash
                ret = p.poll()
                if ret is not None:
                    print(f"[arbiter] Detected crashed pet-server idx={pet_idx}, code={ret}", flush=True)
                    redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
                # 2) React to restart flag
                state = redis_client.get(pet_status_key(pet_idx))
                if state and state.decode() == PetStatus.RESTART_NEEDED:
                    print(f"[arbiter] Restart requested for pet_idx={pet_idx}", flush=True)
                    restart_single_pet_server(pet_idx)
                
                resp = {"id": req_id, "resp": "OK"}
                redis_client.publish(reply_channel, json.dumps(resp))
        except Exception as e:
            print(f"[arbiter] Error in monitor thread pet_idx={pet_idx}: {e}", flush=True)
            time.sleep(1)

    print(f"[arbiter] Monitor thread exiting for pet_idx={pet_idx}", flush=True)


def _shutdown() -> None:
    """Graceful shutdown hook."""
    _stop_event.set()
    stop_pet_servers()
    print("[arbiter] Clear cache state.", flush=True)
    clean_redis_all()


def _signal_handler(signum, frame) -> None:
    print(f"[arbiter] Received signal {signum}, shutting down.", flush=True)
    _shutdown()


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    clean_redis_all()
    kill_all_pet()
    start_pet_servers()

    t = threading.Thread(target=monitor_ram, daemon=True)
    t.start()
    monitor_threads.append(t)
    
    # Start monitor threads
    for pet_idx in range(NUM_PET_SERVER):
        t = threading.Thread(target=monitor_redis_for_restarts, args=(pet_idx,), daemon=True)
        t.start()
        monitor_threads.append(t)

    print("[arbiter] Running.", flush=True)

    # Keep process alive until signal
    try:
        while not _stop_event.is_set():
            time.sleep(1.0)
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
