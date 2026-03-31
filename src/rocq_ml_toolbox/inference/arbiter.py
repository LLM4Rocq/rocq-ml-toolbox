from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from typing import List, Optional

try:
    from setproctitle import setproctitle
except Exception:
    def setproctitle(_: str) -> None:
        return None

setproctitle("rocq-ml-arbiter")

import psutil
import redis
from pytanque import Pytanque, PytanqueMode

from .redis_keys import (
    ALL_KEYS_STAR,
    PetStatus,
    arbiter_heartbeat_key,
    arbiter_key,
    generation_key,
    pet_lock_key,
    pet_status_key,
)

NUM_PET_SERVER = int(os.environ["NUM_PET_SERVER"])
PET_SERVER_START_PORT = int(os.environ["PET_SERVER_START_PORT"])
MAX_RAM_PER_PET = int(os.environ["MAX_RAM_PER_PET"])  # MB; 0 disables RAM checks
REDIS_URL = os.environ["REDIS_URL"]
PET_CMD = os.environ.get("PET_CMD", "pet-server")

PET_READY_TIMEOUT = float(os.environ.get("PET_READY_TIMEOUT", "30.0"))
PET_READY_POLL_INTERVAL = float(os.environ.get("PET_READY_POLL_INTERVAL", "0.2"))
RAM_MONITOR_INTERVAL = float(os.environ.get("RAM_MONITOR_INTERVAL", "0.5"))
MONITOR_POLL_INTERVAL = float(os.environ.get("MONITOR_POLL_INTERVAL", "0.5"))
ARBITER_HEARTBEAT_INTERVAL = float(os.environ.get("ARBITER_HEARTBEAT_INTERVAL", "1.0"))
ARBITER_HEARTBEAT_TTL = int(os.environ.get("ARBITER_HEARTBEAT_TTL", "5"))

redis_client = redis.Redis.from_url(REDIS_URL)

pet_servers: List[Optional[subprocess.Popen]] = [None] * NUM_PET_SERVER
pet_servers_lock = threading.RLock()
monitor_threads: list[threading.Thread] = []

_stop_event = threading.Event()
_shutdown_lock = threading.Lock()
_shutdown_done = False


def _decode(value: bytes | str | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return value


def _set_pet_status(pet_idx: int, status: PetStatus) -> None:
    redis_client.set(pet_status_key(pet_idx), status)


def _get_pet_status(pet_idx: int) -> Optional[str]:
    return _decode(redis_client.get(pet_status_key(pet_idx)))


def _set_arbiter_ready(is_ready: bool) -> None:
    redis_client.set(arbiter_key(), "1" if is_ready else "0")


def _write_heartbeat() -> None:
    redis_client.set(
        arbiter_heartbeat_key(),
        f"{time.time():.6f}",
        ex=max(ARBITER_HEARTBEAT_TTL, 1),
    )


def clean_redis_all() -> None:
    for key in ALL_KEYS_STAR:
        for subkey in redis_client.scan_iter(key):
            redis_client.delete(subkey)


def kill_all_pet(proc_name: str = "pet-server") -> None:
    """Kill all existing pet-server processes (safety on startup)."""
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if name == proc_name or proc_name in cmdline:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(1)


def _probe_pet_server(port: int, timeout: float = 2.0) -> bool:
    """Probe a pet-server using both TCP and a lightweight RPC."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            pass
    except OSError:
        return False

    client = Pytanque("127.0.0.1", port, mode=PytanqueMode.SOCKET)
    try:
        client.connect()
        return True
    except Exception:
        return False
    finally:
        try:
            client.close()
        except Exception:
            pass


def wait_until_pet_ready(pet_idx: int, timeout_s: float = PET_READY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout_s
    port = PET_SERVER_START_PORT + pet_idx
    while time.monotonic() < deadline and not _stop_event.is_set():
        with pet_servers_lock:
            p = pet_servers[pet_idx]
        if p is None:
            return False
        if p.poll() is not None:
            return False
        if _probe_pet_server(port):
            return True
        time.sleep(PET_READY_POLL_INTERVAL)
    return False


def _stop_single_pet_server(pet_idx: int) -> None:
    with pet_servers_lock:
        p = pet_servers[pet_idx]
        pet_servers[pet_idx] = None

    if p is None:
        return

    try:
        os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass

    try:
        p.wait(timeout=3)
    except Exception:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass


def _spawn_single_pet_server(pet_idx: int) -> subprocess.Popen:
    port = PET_SERVER_START_PORT + pet_idx
    p = subprocess.Popen(
        [PET_CMD, "-p", str(port)],
        start_new_session=True,
    )
    with pet_servers_lock:
        pet_servers[pet_idx] = p
    return p


def start_single_pet_server(pet_idx: int) -> bool:
    _set_pet_status(pet_idx, PetStatus.STARTING)
    p = _spawn_single_pet_server(pet_idx)
    if not wait_until_pet_ready(pet_idx):
        print(
            f"[arbiter] pet-server idx={pet_idx} failed to become ready, pid={p.pid}",
            flush=True,
        )
        _stop_single_pet_server(pet_idx)
        _set_pet_status(pet_idx, PetStatus.DOWN)
        return False

    _set_pet_status(pet_idx, PetStatus.OK)
    return True


def start_pet_servers() -> None:
    """Spawn one pet-server process per pet_idx on fixed ports."""
    for pet_idx in range(NUM_PET_SERVER):
        redis_client.set(generation_key(pet_idx), 0)
        if not start_single_pet_server(pet_idx):
            raise RuntimeError(f"failed to start pet-server idx={pet_idx}")

    with pet_servers_lock:
        snapshot = [
            (pet_idx, PET_SERVER_START_PORT + pet_idx, p.pid)
            for pet_idx, p in enumerate(pet_servers)
            if p is not None
        ]
    print("[arbiter] Started pet-servers:", snapshot, flush=True)


def stop_pet_servers() -> None:
    for pet_idx in range(NUM_PET_SERVER):
        _stop_single_pet_server(pet_idx)
        _set_pet_status(pet_idx, PetStatus.DOWN)
    print("[arbiter] Stopped all pet-servers", flush=True)


def restart_single_pet_server(pet_idx: int) -> bool:
    """Restart the pet-server for a single index with generation fencing."""
    _set_pet_status(pet_idx, PetStatus.RESTARTING)

    gen_raw = redis_client.get(generation_key(pet_idx))
    generation = int(_decode(gen_raw) or "0")
    redis_client.set(generation_key(pet_idx), generation + 1)

    _stop_single_pet_server(pet_idx)

    if not start_single_pet_server(pet_idx):
        return False

    with pet_servers_lock:
        p = pet_servers[pet_idx]
    port = PET_SERVER_START_PORT + pet_idx
    print(
        f"[arbiter] Restarted pet-server idx={pet_idx} on port {port}, pid={p.pid if p else None}, gen={generation + 1}",
        flush=True,
    )
    return True


def _is_pet_locked(pet_idx: int) -> bool:
    return bool(redis_client.exists(pet_lock_key(pet_idx)))


def _mark_restart_needed_if_crashed(pet_idx: int) -> None:
    with pet_servers_lock:
        p = pet_servers[pet_idx]
    if p is None:
        _set_pet_status(pet_idx, PetStatus.RESTART_NEEDED)
        return

    ret = p.poll()
    if ret is not None:
        print(
            f"[arbiter] Detected crashed pet-server idx={pet_idx}, code={ret}",
            flush=True,
        )
        _set_pet_status(pet_idx, PetStatus.RESTART_NEEDED)


def _maybe_restart_pet_server(pet_idx: int) -> bool:
    state = _get_pet_status(pet_idx)
    if state != PetStatus.RESTART_NEEDED:
        return False

    print(f"[arbiter] Restart requested for pet_idx={pet_idx}", flush=True)
    return restart_single_pet_server(pet_idx)


def check_ram(pet_idx: int) -> None:
    """Monitor RAM usage and schedule restart when worker exceeds threshold."""
    with pet_servers_lock:
        p = pet_servers[pet_idx]
    if p is None or MAX_RAM_PER_PET <= 0:
        return

    if p.poll() is not None:
        _set_pet_status(pet_idx, PetStatus.RESTART_NEEDED)
        return

    try:
        proc = psutil.Process(p.pid)
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        if rss_mb > MAX_RAM_PER_PET:
            current_status = _get_pet_status(pet_idx)
            if current_status not in (
                PetStatus.RESTART_NEEDED,
                PetStatus.RESTARTING,
            ):
                print(
                    f"[arbiter] pet-server idx={pet_idx} over RAM limit: "
                    f"{rss_mb:.1f} MB > {MAX_RAM_PER_PET} MB; scheduling restart",
                    flush=True,
                )
                _set_pet_status(pet_idx, PetStatus.RESTART_NEEDED)
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        print(
            f"[arbiter] RAM check failed for pet_idx={pet_idx}: {exc}",
            flush=True,
        )
    return

# def monitor_ram(poll_interval: float = RAM_MONITOR_INTERVAL) -> None:
#     """Monitor RAM usage and schedule restart when worker exceeds threshold."""
#     print("[arbiter] RAM monitor started", flush=True)

#     while not _stop_event.is_set():
#         try:
#             for pet_idx in range(NUM_PET_SERVER):
#                 with pet_servers_lock:
#                     p = pet_servers[pet_idx]
#                 if p is None or MAX_RAM_PER_PET <= 0:
#                     continue

#                 if p.poll() is not None:
#                     _set_pet_status(pet_idx, PetStatus.RESTART_NEEDED)
#                     continue

#                 try:
#                     proc = psutil.Process(p.pid)
#                     rss_mb = proc.memory_info().rss / (1024 * 1024)
#                     if rss_mb > MAX_RAM_PER_PET:
#                         current_status = _get_pet_status(pet_idx)
#                         if current_status not in (
#                             PetStatus.RESTART_NEEDED,
#                             PetStatus.RESTARTING,
#                         ):
#                             print(
#                                 f"[arbiter] pet-server idx={pet_idx} over RAM limit: "
#                                 f"{rss_mb:.1f} MB > {MAX_RAM_PER_PET} MB; scheduling restart",
#                                 flush=True,
#                             )
#                             _set_pet_status(pet_idx, PetStatus.RESTART_NEEDED)
#                 except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
#                     print(
#                         f"[arbiter] RAM check failed for pet_idx={pet_idx}: {exc}",
#                         flush=True,
#                     )

#             time.sleep(poll_interval)
#         except Exception as exc:
#             print(f"[arbiter] Error in RAM monitor: {exc}", flush=True)
#             time.sleep(poll_interval)

#     print("[arbiter] RAM monitor exit", flush=True)


def monitor_redis_for_restarts(pet_idx: int) -> None:
    """Monitor restart requests for one worker and handle health replies."""
    print(f"[arbiter] Monitor thread started for pet_idx={pet_idx}", flush=True)
    ps = redis_client.pubsub(ignore_subscribe_messages=True)
    ps.subscribe(f"arbiter:req:{pet_idx}")

    try:
        while not _stop_event.is_set():
            try:
                _mark_restart_needed_if_crashed(pet_idx)
                _maybe_restart_pet_server(pet_idx)

                msg = ps.get_message(timeout=MONITOR_POLL_INTERVAL)
                if not msg or msg.get("type") != "message":
                    continue
                else:
                    check_ram(pet_idx)
                req = json.loads(msg["data"])
                reply_channel = req.get("reply_to")
                req_id = req.get("id")

                _mark_restart_needed_if_crashed(pet_idx)
                _maybe_restart_pet_server(pet_idx)

                state = _get_pet_status(pet_idx)
                generation = int(_decode(redis_client.get(generation_key(pet_idx))) or "0")
                resp = {
                    "id": req_id,
                    "resp": "OK" if state == PetStatus.OK else "NOT_OK",
                    "status": state,
                    "generation": generation,
                }
                if reply_channel:
                    redis_client.publish(reply_channel, json.dumps(resp))
            except Exception as exc:
                print(f"[arbiter] Error in monitor thread pet_idx={pet_idx}: {exc}", flush=True)
                time.sleep(1)
    finally:
        try:
            ps.close()
        except Exception:
            pass

    print(f"[arbiter] Monitor thread exiting for pet_idx={pet_idx}", flush=True)


def _shutdown() -> None:
    """Graceful shutdown hook."""
    global _shutdown_done

    with _shutdown_lock:
        if _shutdown_done:
            return
        _shutdown_done = True

    _stop_event.set()
    _set_arbiter_ready(False)

    try:
        redis_client.delete(arbiter_heartbeat_key())
    except Exception:
        pass

    stop_pet_servers()
    print("[arbiter] Clear cache state.", flush=True)
    clean_redis_all()


def _signal_handler(signum, frame) -> None:
    print(f"[arbiter] Received signal {signum}, shutting down.", flush=True)
    _shutdown()


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _set_arbiter_ready(False)
    clean_redis_all()
    _set_arbiter_ready(False)

    kill_all_pet(proc_name=os.path.basename(PET_CMD))
    start_pet_servers()

    # t = threading.Thread(target=monitor_ram, daemon=True)
    # t.start()
    # monitor_threads.append(t)

    for pet_idx in range(NUM_PET_SERVER):
        t = threading.Thread(
            target=monitor_redis_for_restarts,
            args=(pet_idx,),
            daemon=True,
        )
        t.start()
        monitor_threads.append(t)

    _set_arbiter_ready(True)
    _write_heartbeat()
    print("[arbiter] Running.", flush=True)

    try:
        while not _stop_event.is_set():
            _write_heartbeat()
            time.sleep(ARBITER_HEARTBEAT_INTERVAL)
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
