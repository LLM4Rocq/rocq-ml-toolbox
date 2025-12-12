# gunicorn_config.py
import subprocess
import time
import os
import threading
from typing import List, Optional
import psutil

import redis

from inference_server.redis_keys import monitor_epoch_key, pet_status_key, generation_key, cache_state_key, PetStatus

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

NUM_PET_SERVER = int(os.environ.get("NUM_PET_SERVER", 8))
PET_SERVER_START_PORT = int(os.environ.get("PET_SERVER_START_PORT", 8765))
# Maximum allowed ram usage in MB per pet-server process.
MAX_RAM_PER_PET = int(os.environ.get("MAX_RAM_PER_PET", 3072))
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Redis client

redis_client = redis.Redis.from_url(REDIS_URL)

# Track Popen objects for each pet_idx
pet_servers: List[Optional[subprocess.Popen]] = [None] * NUM_PET_SERVER
monitor_threads = []

# ------------------------------------------------------------------------------
# Pet-server lifecycle management
# ------------------------------------------------------------------------------

def start_pet_servers():
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
        [(pet_idx, PET_SERVER_START_PORT + pet_idx, p.pid) for pet_idx, p in enumerate(pet_servers)],
        flush=True,
    )

def stop_pet_servers():
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

def restart_single_pet_server(pet_idx: int):
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
    
    generation = int(redis_client.get(generation_key(pet_idx)))
    redis_client.set(generation_key(pet_idx), generation + 1)
    time.sleep(3.0)
    # Mark as OK again
    redis_client.set(pet_status_key(pet_idx), PetStatus.OK)
    print(f"[arbiter] Restarted pet-server idx={pet_idx} on port {port}, pid={new_p.pid}", flush=True)

# ------------------------------------------------------------------------------
# Monitor Redis for restart requests
# ------------------------------------------------------------------------------

def monitor_redis_for_restarts(pet_idx: int, poll_interval: float = 0.02):
    """
    Monitor Redis pet_status:{idx} keys for RESTART_NEEDED, detect crashes,
    and restart the corresponding pet-server. Also monitor RAM usage and
    trigger restart if it exceeds MAX_RAM_PER_PET (MB).
    """
    print("[arbiter] Redis restart monitor started", flush=True)
    epoch_key = monitor_epoch_key(pet_idx)

    while True:
        try:
            p = pet_servers[pet_idx]
            if p is None:
                time.sleep(poll_interval)
                continue

            # 1) Detect if process died without a RESTART_NEEDED flag.
            ret = p.poll()
            if ret is not None:
                # Process is dead but Redis might still say OK.
                print(f"[arbiter] Detected crashed pet-server idx={pet_idx}, code={ret}", flush=True)
                redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
            else:
                # 2) Process is alive: check RAM usage if threshold enabled
                if MAX_RAM_PER_PET > 0:
                    try:
                        proc = psutil.Process(p.pid)
                        rss_bytes = proc.memory_info().rss
                        rss_mb = rss_bytes / (1024 * 1024)
                        if rss_mb > MAX_RAM_PER_PET:
                            print(
                                f"[arbiter] pet-server idx={pet_idx} over RAM limit: "
                                f"{rss_mb:.1f} MB > {MAX_RAM_PER_PET} MB; scheduling restart",
                                flush=True,
                            )
                            redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        print(
                            f"[arbiter] RAM check failed for pet_idx={pet_idx}: {e}",
                            flush=True,
                        )

            # 3) React to RESTART_NEEDED
            state = redis_client.get(pet_status_key(pet_idx))
            if state:
                decoded = state.decode()
                if decoded == PetStatus.RESTART_NEEDED:
                    print(f"[arbiter] Restart requested for pet_idx={pet_idx}", flush=True)

                    # Mark as RESTARTING to tell workers not to use this pet
                    redis_client.set(pet_status_key(pet_idx), PetStatus.RESTARTING)
                    restart_single_pet_server(pet_idx)

            # 4) One full iteration completed: bump monitor epoch
            redis_client.incr(epoch_key, amount=1)

            time.sleep(poll_interval)
        except Exception as e:
            print(f"[arbiter] Error in Redis restart monitor: {e}", flush=True)
            time.sleep(poll_interval)

# ------------------------------------------------------------------------------
# Gunicorn hooks
# ------------------------------------------------------------------------------

def on_starting(server):
    """Called just before the master process is initialized."""
    global monitor_threads
    start_pet_servers()

    # Start background monitor thread in arbiter
    for pet_idx in range(NUM_PET_SERVER):
        monitor_thread = threading.Thread(
            target=monitor_redis_for_restarts,
            args=(pet_idx,),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

def on_exit(server):
    """Called just before exiting Gunicorn master process."""
    stop_pet_servers()
    print(f"[arbiter] Clear cache state.", flush=True)
    for key in redis_client.scan_iter("cache_state:*"):
        redis_client.delete(key)
