from __future__ import annotations

import argparse
import os
from typing import List, Optional

DEFAULT_APP = "rocq_ml_toolbox.inference.server:app"
DEFAULT_CONFIG = "python:rocq_ml_toolbox.inference.gunicorn_config"

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="rocq-ml-server")
    p.add_argument("-H", "--host", default="0.0.0.0")
    p.add_argument("-p", "--port", type=int, default=5000)
    p.add_argument("-w", "--workers", type=int, default=9)
    p.add_argument("-t", "--timeout", type=int, default=600)
    p.add_argument("-d", "--detached", action="store_true", help="Run gunicorn in the background (daemon mode).")
    p.add_argument("--pidfile", default="rocq-ml-server.pid", help="PID file (with --detached).")
    p.add_argument("--num-pet-server", type=int, default=4)
    p.add_argument("--pet-server-start-port", type=int, default=8765)
    p.add_argument("--max-ram-per-pet", type=int, default=3072, help="Maximum allowed ram usage in MB per pet-server process.")
    p.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")

    p.add_argument("--errorlog", default="gunicorn-error.log")
    p.add_argument("--accesslog", default="gunicorn-access.log")

    p.add_argument("--app", default=DEFAULT_APP)
    p.add_argument("--config", default=DEFAULT_CONFIG)

    args = p.parse_args(argv)

    cmd = [
        "gunicorn",
        args.app,
        "-b", f"{args.host}:{args.port}",
        "-c", args.config,
        "-w", str(args.workers),
        "-t", str(args.timeout),
        "--error-logfile", args.errorlog,
        "--access-logfile", args.accesslog,
        "--capture-output",
    ]
    if args.detached:
        cmd += ["--daemon", "--pid", args.pidfile]

    os.environ["NUM_PET_SERVER"] = str(args.num_pet_server)
    os.environ["PET_SERVER_START_PORT"] = str(args.pet_server_start_port)
    os.environ["MAX_RAM_PER_PET"] = str(args.max_ram_per_pet)
    os.environ["REDIS_URL"] = str(args.redis_url)
    os.execvp("gunicorn", cmd)
