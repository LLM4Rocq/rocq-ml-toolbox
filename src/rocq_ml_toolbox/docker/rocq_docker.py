"""Helpers to manage Docker containers for OPAM-based Coq environments."""

import os
import time, socket
from pathlib import Path
from typing import Dict, Any
import re
import shlex
import sys
import signal

import requests
import docker
from docker.models.containers import Container

from .config import OpamConfig
from ..parser.parser import Source

class RocqDocker:
    """Wraps Docker interactions for extracting data from an OPAM switch."""

    @staticmethod
    def handler(signum, frame):
        """Signal handler for timeouts."""
        raise TimeoutError("Operation timed out")

    def __init__(self, config:OpamConfig, redis_image: str= "redis:latest", redis_port:int = 6379, rebuild=False, kill_clone=False):
        """Start or reuse a container built from the given OPAM configuration."""
        super().__init__()
        self.client = docker.from_env()
        self.config = config
        self._build_image(config, rebuild=rebuild)
        image_name = config.name + ':' + config.tag

        if kill_clone:
            try:
                for c in self.client.containers.list(all=True, filters={"ancestor": image_name}):
                    c.reload()
                    if c.status == "running":
                        try:
                            c.kill()
                        except Exception:
                            pass
            except Exception as e:
                raise RuntimeError(f"Failed to kill clones for image {image_name}: {e}")
            try:
                for c in self.client.containers.list(all=True, filters={"ancestor": redis_image}):
                    c.reload()
                    if c.status == "running":
                        try:
                            c.kill()
                        except Exception:
                            pass
            except Exception as e:
                raise RuntimeError(f"Failed to kill clones for image {redis_image}: {e}")
        self.container = self.client.containers.run(
            image_name,
            detach=True,
            tty=False,
            stdin_open=False,
            user=config.user,
            command=["sleep", "infinity"],
            network_mode="host",
        )
        self.redis_container = self.client.containers.run(
            redis_image,
            detach=True,
            ports={"6379/tcp": ("127.0.0.1", redis_port)},
            restart_policy={"Name": "unless-stopped"}
        )
        self.opam_env_path = config.opam_env_path
    
    def _build_image(self, config:OpamConfig, rebuild=False, timeout_install=3600):
        """Create a container image for the requested packages if needed."""
        new_image_name = config.name + ":" + config.tag
        filterred_images = self.client.images.list(filters={'reference': new_image_name})
        if filterred_images and not rebuild:
            print('Image already exists')
            return
        
        self.container = self.client.containers.run(
            config.base_image,
            detach=True,
            tty=False,
            stdin_open=False,
            user=config.user,
            command=["sleep", "infinity"],
            network_mode="host",
        )
        try:
            print('Build Image')
            signal.signal(signal.SIGALRM, RocqDocker.handler)
            signal.alarm(timeout_install)
            self.install_project(" ".join(config.packages))
            signal.alarm(0)
        except Exception as e:
            print(f"WARNING: {e}")
        self.container.commit(config.name, config.tag)
        self.kill_container(self.container)

    def _ensure_running(self):
        """Guarantee the container is running before issuing commands."""
        self.container.reload()
        if self.container.status != "running":
            self.container.start()
            self.container.reload()
            if self.container.status != "running":
                raise RuntimeError(f"Container not running (status={self.container.status})")

    def _stream_exec(self, cmd: str, *, demux: bool = True) -> int:
        """Run a command inside the container and stream its output."""
        self._ensure_running()
        api = self.client.api
        exec_id = api.exec_create(
            self.container.id,
            cmd,
            stdout=True,
            stderr=True,
            tty=False,
        )["Id"]

        for chunk in api.exec_start(exec_id, stream=True, demux=demux):
            if chunk is None:
                continue
            if demux:
                out, err = chunk
                if out:
                    sys.stdout.write(out.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                if err:
                    sys.stderr.write(err.decode("utf-8", errors="replace"))
                    sys.stderr.flush()
            else:
                if isinstance(chunk, (bytes, bytearray)):
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()

        status = api.exec_inspect(exec_id)
        return int(status.get("ExitCode", 1))

    def install_project(self, project: str, extra_args: str = ""):
        """Install OPAM packages inside the container image."""
        env = "OPAMYES=1 OPAMCOLOR=never"
        cmd_install = f"{env} opam install {project} {extra_args}"
        cmd = f"sh -lc '{cmd_install}'"
        code = self._stream_exec(cmd, demux=True)
        if code != 0:
            raise RuntimeError(f"`opam install {project}` failed with exit code {code}")

    def exec_cmd(self, cmd) -> str:
        """Execute a command without streaming and return its stdout."""
        api = self.client.api
        exec_id = api.exec_create(
            self.container.id,
            cmd,
            stdout=True,
            stderr=True,
            tty=False,
        )["Id"]
        return self.client.api.exec_start(exec_id).decode('utf-8')

    def _read_file(self, filepath, max_bytes=None, encoding="utf-8") -> str:
        """Read a file from the container filesystem."""
        api = self.client.api
        cmd = f"sh -lc 'cat -- {shlex.quote(filepath)}'"
        exec_id = api.exec_create(self.container.id, cmd,
                                stdout=True, stderr=True, tty=False)["Id"]

        buf = bytearray()
        for chunk in api.exec_start(exec_id, stream=True, demux=True):
            if chunk is None:
                continue
            out, err = chunk
            if err:
                raise RuntimeError(err.decode("utf-8", "replace"))
            if out:
                buf.extend(out)
                if max_bytes and len(buf) >= max_bytes:
                    break

        code = api.exec_inspect(exec_id).get("ExitCode", 1)
        if code != 0 and not buf:
            raise RuntimeError(f"cat failed with exit code {code}")
        return buf.decode(encoding, errors="replace")

    def kill_container(self, container: Container):
        try:
            container.kill()
        finally:
            container.remove(force=True)

    def close(self):
        """Stop and remove the underlying containers."""
        self.kill_container(self.container)
        self.kill_container(self.redis_container)

    def start_inference_server(self, port=5000, workers=9, timeout=600, num_pet_server=4, pet_server_start_port=8765, max_ram_per_pet=3072):
        """Launch pet-server inside the container."""
        self.pet_port = port
        cmd = f"""
        eval "$(/home/rocq/miniconda/bin/conda shell.bash hook)"
        conda activate rocq-ml
        nohup rocq-ml-server -p {port} -w {workers} -t {timeout} \
        --num-pet-server {num_pet_server} \
        --pet-server-start-port {pet_server_start_port} \
        --max-ram-per-pet {max_ram_per_pet} &
        """

        self.exec_cmd(["bash", "-lc", cmd])
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health")
                if resp.status_code == 200:
                    return
            except Exception as e:
                pass
            time.sleep(0.1)
        log = self.exec_cmd("sh -lc 'tail -n +200 /tmp/gunicorn-error.log || true'")
        raise RuntimeError(f"pet-server failed to start on port {port}.\n{log}")

    def extract_opam_path(self, package_name: str, info_path: Dict[str, str]):
        """Resolve the OPAM installation path for a package."""
        opam_show = self.exec_cmd(f"opam show {package_name}")
        match = re.search(r'"logpath:([A-Za-z0-9_.-]+)"', opam_show)
        if not match:
            assert package_name in info_path and info_path[package_name], f'Missing info_path for {package_name}'
            return info_path[package_name], opam_show
        return match.group(1), opam_show

    def extract_files(self, package_name: str, info_path: Dict[str, str]) -> Dict[str, Any]:
        """List `.v` files shipped with an installed package."""
        fqn, opam_show = self.extract_opam_path(package_name, info_path)
        user_contrib_path = os.path.join(self.opam_env_path, "lib/coq/user-contrib/")
        sources_path = os.path.join(user_contrib_path, fqn.replace('.', '/'))
        subfiles = self.exec_cmd(f"find {sources_path}").splitlines()
        return {"fqn": fqn, "package_name": package_name, "root": user_contrib_path, "subfiles": [file for file in subfiles if file.endswith('.v')], "opam_show": opam_show}

    def get_source(self, filepath) -> Source:
        """Return a `Source` dataclass for a file inside the container."""
        content = self._read_file(filepath)
        return Source(path=Path(filepath), content=content)
    
