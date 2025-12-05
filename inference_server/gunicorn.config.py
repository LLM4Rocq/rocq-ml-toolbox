# gunicorn_config.py
import subprocess
import yaml
import time
import os
import threading

import redis

# ------------------------------------------------------------------------------
# Redis client and helpers
# ------------------------------------------------------------------------------

redis_client = redis.Redis.from_url(
    os.environ.get("REDIS_URL", "redis://localhost:6379/0")
)

def pet_status_key(pet_idx: int) -> str:
    return f"pet_status:{pet_idx}"

def current_generation(pet_idx: int) -> str:
    return f"generation:{str(pet_idx)}"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

CONFIG_PATH = os.environ.get("PET_CONFIG_PATH", "config/server/script/config.yaml")
NUM_PET_SERVER = int(os.environ.get("NUM_PET_SERVER", 4))
PET_SERVER_START_PORT = int(os.environ.get("PET_SERVER_START_PORT", 8765))

# Track Popen objects for each pet_idx
pet_servers = [None] * NUM_PET_SERVER
monitor_thread = None

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
        redis_client.set(pet_status_key(pet_idx), "OK")
        redis_client.set(current_generation(pet_idx), 0)
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
        # Optionally mark as DOWN; your SessionManager currently expects OK/RESTART_NEEDED
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
    
    generation = int(redis_client.get(current_generation(pet_idx)))
    redis_client.set(current_generation(pet_idx), generation + 1)
    time.sleep(3.0)
    # Mark as OK again
    redis_client.set(pet_status_key(pet_idx), "OK")
    print(f"[arbiter] Restarted pet-server idx={pet_idx} on port {port}, pid={new_p.pid}", flush=True)

# ------------------------------------------------------------------------------
# Monitor Redis for restart requests
# ------------------------------------------------------------------------------

def monitor_redis_for_restarts(poll_interval: float = 0.1):
    """
    Monitor Redis pet_status:{idx} keys for RESTART_NEEDED and restart
    the corresponding pet-server.
    """
    print("[arbiter] Redis restart monitor started", flush=True)

    while True:
        try:
            for pet_idx in range(NUM_PET_SERVER):
                state = redis_client.get(pet_status_key(pet_idx))
                if not state:
                    continue
                decoded = state.decode()

                if decoded == "RESTART_NEEDED":
                    print(f"[arbiter] Restart requested for pet_idx={pet_idx}", flush=True)

                    # Mark as RESTARTING to tell workers not to use this pet
                    redis_client.set(pet_status_key(pet_idx), "RESTARTING")

                    # Perform the actual restart
                    restart_single_pet_server(pet_idx)

            time.sleep(poll_interval)
        except Exception as e:
            print(f"[arbiter] Error in Redis restart monitor: {e}", flush=True)
            time.sleep(1.0)

# ------------------------------------------------------------------------------
# Gunicorn hooks
# ------------------------------------------------------------------------------

def on_starting(server):
    """Called just before the master process is initialized."""
    global monitor_thread
    start_pet_servers()

    # Start background monitor thread in arbiter
    monitor_thread = threading.Thread(
        target=monitor_redis_for_restarts,
        daemon=True,
    )
    monitor_thread.start()

def on_exit(server):
    """Called just before exiting Gunicorn master process."""
    stop_pet_servers()
