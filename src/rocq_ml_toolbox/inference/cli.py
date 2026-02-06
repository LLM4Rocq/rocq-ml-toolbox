from __future__ import annotations

import argparse
import os
import sys
import subprocess
from typing import List, Optional
from pathlib import Path

DEFAULT_APP = "rocq_ml_toolbox.inference.server:app"
DEFAULT_CONFIG = "python:rocq_ml_toolbox.inference.gunicorn_config"

def popen_detached(cmd, env, pidfile: str | None = None):
    with open(os.devnull, "rb") as devnull_in, open(os.devnull, "wb") as devnull_out:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=devnull_out,
            start_new_session=True
        )

    if pidfile:
        Path(pidfile).write_text(str(proc.pid))

    return proc

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="rocq-ml-server")
    p.add_argument("-H", "--host", default="0.0.0.0")
    p.add_argument("-d", "--detached", action="store_true", help="Run server in background (detach from terminal).")
    p.add_argument("-p", "--port", type=int, default=5000)
    p.add_argument("-w", "--workers", type=int, default=21)
    p.add_argument("-t", "--timeout", type=int, default=600)
    p.add_argument("-l", "--log", action="store_true", default=False)
    p.add_argument("--pidfile", default="rocq-ml-server.pid", help="PID file (with --detached).")
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
    
    print("Starting arbiter...")
    arbiter_proc = popen_detached(
        arbiter_cmd,
        env=env,
        pidfile=args.pidfile + ".arbiter" if args.detached else None,
    )

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
            arbiter_proc.terminate()
            try:
                arbiter_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                arbiter_proc.kill()
