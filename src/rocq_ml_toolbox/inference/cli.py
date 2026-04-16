from __future__ import annotations

import argparse
import os
import sys
import subprocess
import redis
import uuid
import json
import socket
import signal
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

    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=devnull_in,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )
    finally:
        devnull_in.close()
        stdout_f.close()
        stderr_f.close()

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
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=timeout_s)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


def redis_url_from_port(port: int) -> str:
    return f"redis://127.0.0.1:{port}/0"

def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def wait_for_redis(redis_client: redis.Redis, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if redis_client.ping():
                return True
        except redis.ConnectionError:
            pass
        time.sleep(0.2)
    return False


def restart_redis_server(redis_client: redis.Redis, port: int) -> subprocess.Popen:
    try:
        redis_client.shutdown(nosave=True)
    except redis.ConnectionError:
        pass
    except redis.RedisError:
        # Ignore shutdown failures and attempt a fresh start.
        pass

    proc = subprocess.Popen(
        ["redis-server", "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="rocq-ml-server")
    p.add_argument("-H", "--host", default="0.0.0.0")
    p.add_argument("-d", "--detached", action="store_true", help="Run server in background (detach from terminal).")
    p.add_argument("-p", "--port", type=int, default=5000)
    p.add_argument("-w", "--workers", type=int, default=9)
    p.add_argument("-t", "--timeout", type=int, default=600)
    p.add_argument("-l", "--log", action="store_true", default=False)
    p.add_argument("--pidfile", default="rocq-ml-server.pid", help="PID file (with --detached).")
    p.add_argument("--arbiter-log", default="arbiter.log", help="arbiter log file.")
    p.add_argument("--num-pet-server", type=int, default=4)
    p.add_argument("--pet-server-start-port", type=int, default=8765)
    p.add_argument("--pet-server-cmd", type=str, default="pet-server")
    p.add_argument("--soft-max-ram-per-pet", type=int, default=4000, help="Maximum allowed ram usage in MB per pet-server process (soft interruption).")
    p.add_argument("--hard-max-ram-per-pet", type=int, default=6000, help="Maximum allowed ram usage in MB per pet-server process (hard interruption).")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--session-ttl-seconds",
        type=int,
        default=int(os.environ.get("SESSION_TTL_SECONDS", str(10*60*60))),
        help="Session inactivity TTL in seconds before eviction.",
    )
    p.add_argument(
        "--session-cache-keep-feedback",
        action="store_true",
        default=env_bool("SESSION_CACHE_KEEP_FEEDBACK", default=False),
        help="Keep State.feedback in server-side caches (disabled by default).",
    )
    p.add_argument(
        "--session-cleanup-interval-seconds",
        type=int,
        default=int(os.environ.get("SESSION_CLEANUP_INTERVAL_SECONDS", "60")),
        help="How often to scan and evict expired sessions.",
    )
    p.add_argument(
        "--fs-access-mode",
        choices=["read_lib_only", "rw_anywhere"],
        default=os.environ.get("FS_ACCESS_MODE", "read_lib_only"),
        help="Filesystem access policy for read_file/write_file endpoints (immutable at startup).",
    )
    p.add_argument(
        "--coq-lib-path",
        default=os.environ.get("COQ_LIB_PATH"),
        help="Optional override for Coq lib root (defaults to `coqc -where`).",
    )
    p.add_argument(
        "--fs-read-allow",
        action="append",
        default=None,
        help="Additional read-allowed root path (repeatable) in read_lib_only mode.",
    )
    p.add_argument("--app", default=DEFAULT_APP)
    p.add_argument("--config", default=DEFAULT_CONFIG)

    args = p.parse_args(argv)
    redis_url = redis_url_from_port(args.redis_port)

    if args.fs_read_allow is not None:
        fs_read_allow_paths = list(args.fs_read_allow)
    else:
        env_paths_raw = os.environ.get("FS_READ_ALLOW_PATHS", "[]")
        try:
            parsed = json.loads(env_paths_raw)
        except Exception:
            parsed = []
        fs_read_allow_paths = [str(p) for p in parsed] if isinstance(parsed, list) else []
    
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
    env["SOFT_MAX_RAM_PER_PET"] = str(args.soft_max_ram_per_pet)
    env["HARD_MAX_RAM_PER_PET"] = str(args.hard_max_ram_per_pet)
    env["REDIS_URL"] = redis_url
    env["PET_CMD"] = str(args.pet_server_cmd)
    env["SESSION_TTL_SECONDS"] = str(max(0, int(args.session_ttl_seconds)))
    env["SESSION_CACHE_KEEP_FEEDBACK"] = "1" if args.session_cache_keep_feedback else "0"
    env["SESSION_CLEANUP_INTERVAL_SECONDS"] = str(max(1, int(args.session_cleanup_interval_seconds)))
    env["FS_ACCESS_MODE"] = str(args.fs_access_mode)
    if args.coq_lib_path:
        env["COQ_LIB_PATH"] = str(args.coq_lib_path)
    env["FS_READ_ALLOW_PATHS"] = json.dumps(fs_read_allow_paths)
    
    first_pet_port = args.pet_server_start_port
    all_required_ports = list(range(first_pet_port, first_pet_port+ args.num_pet_server))
    all_required_ports.append(args.port)

    for port in all_required_ports:
        if not is_port_available(port):
            raise OSError(f"Required port {port} for pet-server is already in use on localhost. Please use --pet-server-start-port to point to available ports.")
    
    print("Starting redis...")
    redis_proc: subprocess.Popen | None = None
    redis_client = redis.Redis.from_url(redis_url)
    try:
        redis_proc = restart_redis_server(redis_client, args.redis_port)
    except FileNotFoundError as exc:
        raise RuntimeError("redis-server executable not found in PATH.") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to start redis-server on port {args.redis_port}: {exc}") from exc

    if not wait_for_redis(redis_client, timeout_s=15.0):
        if redis_proc is not None:
            terminate_process(redis_proc)
        raise RuntimeError(f"Redis did not become ready on port {args.redis_port}")
    redis_client.set(arbiter_key(), "0")

    print("Starting arbiter...")
    
    arbiter_log = args.arbiter_log
    arbiter_proc: subprocess.Popen | None = None
    try:
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
                raise TimeoutError(
                    f"Pet-server at {pet_idx} is not ready.\n"
                    f"Arbiter return code: {arbiter_proc.poll()}\n"
                    f"Arbiter log tail:\n{tail(arbiter_log)}"
                )

        print("Starting uvicorn...")
        if args.detached:
            uvicorn_proc = popen_detached(
                uvicorn_cmd,
                env=env,
                pidfile=args.pidfile,
            )
            if redis_proc is not None:
                Path(args.pidfile + ".redis").write_text(str(redis_proc.pid))
            Path(args.pidfile + ".uvicorn").write_text(str(uvicorn_proc.pid))
            return

        uvicorn_proc: subprocess.Popen | None = None
        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        def _raise_interrupt(signum, frame):
            del signum, frame
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _raise_interrupt)
        signal.signal(signal.SIGTERM, _raise_interrupt)
        try:
            uvicorn_proc = subprocess.Popen(uvicorn_cmd, env=env, start_new_session=True)
            while True:
                uvicorn_rc = uvicorn_proc.poll()
                arbiter_rc = arbiter_proc.poll()
                if arbiter_rc is not None:
                    raise RuntimeError(f"arbiter exited unexpectedly with code {arbiter_rc}")
                if uvicorn_rc is not None:
                    if uvicorn_rc != 0:
                        raise subprocess.CalledProcessError(uvicorn_rc, uvicorn_cmd)
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopping server and arbiter...")
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            if uvicorn_proc is not None:
                terminate_process(uvicorn_proc)
            terminate_process(arbiter_proc)
            if redis_proc is not None:
                terminate_process(redis_proc)
    except Exception:
        if arbiter_proc is not None:
            terminate_process(arbiter_proc)
        if redis_proc is not None:
            terminate_process(redis_proc)
        raise
