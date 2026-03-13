from __future__ import annotations

import argparse
import os
import sys
import subprocess
import redis
import uuid
import json
import socket
from typing import List, Optional
from pathlib import Path
import time
try:
    from setproctitle import setproctitle
except Exception:
    def setproctitle(_: str) -> None:
        return None

setproctitle("rocq-ml-server")

from .redis_keys import PetStatus, arbiter_key
DEFAULT_APP = "rocq_ml_toolbox.inference.server:app"
DEFAULT_CONFIG = "python:rocq_ml_toolbox.inference.gunicorn_config"


def popen_detached(cmd, env, pidfile: str | None = None, *, stdout_path=None, stderr_path=None):
    stdout_f = open(stdout_path, "ab") if stdout_path else open(os.devnull, "wb")
    stderr_f = open(stderr_path, "ab") if stderr_path else open(os.devnull, "wb")
    devnull_in = open(os.devnull, "rb")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdin=devnull_in,
        stdout=stdout_f,
        stderr=stderr_f,
        start_new_session=True,
    )

    if pidfile:
        Path(pidfile).write_text(str(proc.pid))

    return proc

def tail(path: str, n: int = 80) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except FileNotFoundError:
        return "<no log file>"

def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def terminate_process(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=timeout_s)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="rocq-ml-server")
    p.add_argument("-H", "--host", default="0.0.0.0")
    p.add_argument("-d", "--detached", action="store_true", help="Run server in background (detach from terminal).")
    p.add_argument("-p", "--port", type=int, default=5000)
    p.add_argument("-w", "--workers", type=int, default=21)
    p.add_argument("-t", "--timeout", type=int, default=600)
    p.add_argument("-l", "--log", action="store_true", default=False)
    p.add_argument("--pidfile", default="rocq-ml-server.pid", help="PID file (with --detached).")
    p.add_argument("--arbiter-log", default="arbiter.log", help="arbiter log file.")
    p.add_argument("--num-pet-server", type=int, default=4)
    p.add_argument("--pet-server-start-port", type=int, default=8765)
    p.add_argument("--pet-server-cmd", type=str, default="pet-server")
    p.add_argument("--max-ram-per-pet", type=int, default=6000, help="Maximum allowed ram usage in MB per pet-server process.")
    p.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    p.add_argument("--app", default=DEFAULT_APP)
    p.add_argument("--config", default=DEFAULT_CONFIG)

    args = p.parse_args(argv)
    
    uvicorn_cmd = [
        "uvicorn",
        "rocq_ml_toolbox.inference.server:app",
        "--host", args.host,
        "--port", str(args.port),
        "--workers", str(args.workers),
        "--timeout-worker-healthcheck", str(args.timeout)
    ]
    if args.log:
        LOG_CFG = Path(__file__).resolve().parent / "logging_config.yaml"
        uvicorn_cmd.extend(["--log-config", str(LOG_CFG)])
    
    arbiter_cmd = [sys.executable, "-m", "rocq_ml_toolbox.inference.arbiter"]
    env = os.environ.copy()
    env["NUM_PET_SERVER"] = str(args.num_pet_server)
    env["PET_SERVER_START_PORT"] = str(args.pet_server_start_port)
    env["MAX_RAM_PER_PET"] = str(args.max_ram_per_pet)
    env["REDIS_URL"] = str(args.redis_url)
    env["PET_CMD"] = str(args.pet_server_cmd)    
    
    first_pet_port = args.pet_server_start_port
    all_required_ports = list(range(first_pet_port, first_pet_port+ args.num_pet_server))
    all_required_ports.append(args.port)

    for port in all_required_ports:
        if not is_port_available(port):
            raise OSError(f"Required port {port} is already in use on localhost.")
    
    print("Starting arbiter...")
    try:
        redis_client = redis.Redis.from_url(args.redis_url)
        redis_client.set(arbiter_key(), "0")
    except redis.ConnectionError:
        raise Exception(f'Redis is not available at {args.redis_url}')
    
    arbiter_log = args.arbiter_log
    arbiter_proc = popen_detached(
        arbiter_cmd,
        env=env,
        pidfile=args.pidfile + ".arbiter" if args.detached else None,
        stdout_path=arbiter_log,
        stderr_path=arbiter_log,
    )
    deadline = time.monotonic() + 60
    arbiter_ready = False
    while time.monotonic() < deadline:
        if arbiter_proc.poll() is not None:
            break
        res = redis_client.get(arbiter_key())
        if res and int(res) == 1:
            arbiter_ready = True
            break
        time.sleep(0.2)
    
    if not arbiter_ready:
        terminate_process(arbiter_proc)
        raise TimeoutError(
            "Arbiter does not respond.\n"
            f"Arbiter return code: {arbiter_proc.poll()}\n"
            f"Arbiter log tail:\n{tail(arbiter_log)}"
        )

    for pet_idx in range(args.num_pet_server):
        req_id = str(uuid.uuid4())
        reply_channel = f"arbiter:reply:{pet_idx}:{req_id}"
        ps = redis_client.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(reply_channel)
        try:
            req = {"id": req_id, "reply_to": reply_channel}
            redis_client.publish(f"arbiter:req:{pet_idx}", json.dumps(req))
            deadline = time.monotonic() + 60
            pet_is_ok = False
            while time.monotonic() < deadline:
                msg = ps.get_message(timeout=1.0)
                if not msg:
                    continue
                if msg["type"] != "message":
                    continue
                resp = json.loads(msg["data"])
                if resp.get("id") != req_id:
                    continue
                pet_is_ok = (
                    resp.get("resp") == "OK" and resp.get("status") == PetStatus.OK
                )
                break
        finally:
            try:
                ps.unsubscribe(reply_channel)
            finally:
                ps.close()

        if not pet_is_ok:
            terminate_process(arbiter_proc)
            raise TimeoutError(
                f"Pet-server at {pet_idx} is not ready.\n"
                f"Arbiter return code: {arbiter_proc.poll()}\n"
                f"Arbiter log tail:\n{tail(arbiter_log)}"
            )

    print("Starting uvicorn...")
    if args.detached:
        popen_detached(
            uvicorn_cmd,
            env=env,
            pidfile=args.pidfile,
        )
    else:
        try:
            subprocess.run(uvicorn_cmd, env=env, check=True)
        except KeyboardInterrupt:
            print("\nStopping server and arbiter...")
        finally:
            # Ensure the arbiter is killed when the server stops
            terminate_process(arbiter_proc)
