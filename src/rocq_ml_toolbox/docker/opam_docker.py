"""Helpers to manage Docker containers for OPAM-based Coq environments."""

import os
from pathlib import Path
from typing import Dict, Any, List
import re
import time
import signal
import tempfile

import requests

from .config import OpamConfig
from .docker import BaseDocker
from ..parser.parser import Source

class OpamDocker(BaseDocker):
    """Wraps Docker interactions for extracting data from an OPAM switch."""

    def __init__(self, config:OpamConfig, redis_image: str= "redis:latest", redis_port:int = 6379, rebuild=False, kill_clone=True, update_rocq_ml=False):
        """Start or reuse a container built from the given OPAM configuration."""
        super().__init__(config, kill_clone=kill_clone, rebuild=rebuild)
        if kill_clone: self._kill_clone(redis_image)
        self.redis_container = self.client.containers.run(
            redis_image,
            detach=True,
            ports={"6379/tcp": ("127.0.0.1", redis_port)},
            restart_policy={"Name": "unless-stopped"}
        )
        self.opam_env_path = config.opam_env_path
        self.config = config
        if update_rocq_ml:
            self.exec_cmd([
                "bash",
                "-lc",
                "cd ~/rocq-ml-toolbox && git pull"
            ])


    def install_project(self, project: str, extra_args: str = ""):
        """Install OPAM packages inside the container image."""
        env = "OPAMYES=1 OPAMCOLOR=never"
        cmd_install = f"{env} opam install {project} {extra_args}"
        cmd = f"sh -lc '{cmd_install}'"
        code = self._stream_exec(cmd, demux=True)
        if code != 0:
            raise RuntimeError(f"`opam install {project}` failed with exit code {code}")

    def close(self):
        """Stop and remove the underlying containers."""
        self.kill_container(self.redis_container)
        super().close()

    def pin_project(self, text: str):
        cmd = f"sh -lc 'opam pin -y {text}'"
        code = self._stream_exec(cmd, demux=True)
        if code != 0:
            raise RuntimeError(f"`opam pin -y {text}` failed with exit code {code}")
    
    def _build_image(self, timeout_install=3600):
        """Create a container image for the requested packages if needed."""
        self.container = self.client.containers.run(
            self.config.base_image,
            detach=True,
            tty=False,
            stdin_open=False,
            user=self.config.user,
            command=["sleep", "infinity"],
            network_mode="host",
        )

        print('Build Image')
        signal.signal(signal.SIGALRM, BaseDocker._timeout_handler)
        signal.alarm(timeout_install)
        if self.config.pins:
            self.pin_project(" ".join(self.config.pins))
        if self.config.packages:
            self.install_project(" ".join(self.config.packages))
        signal.alarm(0)

        self.container.commit(self.config.name, self.config.tag)

    def start_inference_server(self, port=5000, workers=9, timeout=600, num_pet_server=4, pet_server_start_port=8765, max_ram_per_pet=3072):
        """Launch pet-server inside the container."""
        self.pet_port = port
        cmd = f"""
        eval "$(/home/{self.config.user}/miniconda/bin/conda shell.bash hook)"
        conda activate rocq-ml
        rocq-ml-server -d -p {port} -w {workers} -t {timeout} \
        --num-pet-server {num_pet_server} \
        --pet-server-start-port {pet_server_start_port} \
        --max-ram-per-pet {max_ram_per_pet}
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
        raise RuntimeError(f"rocq-ml-server failed to start on port {port}.\n{log}")

    def list_opam_folder(self) -> List[str]:
        "Extract all opam folder"
        user_contrib = os.path.join(self.opam_env_path, "lib/coq/user-contrib/")
        subfiles = self.exec_cmd(f"ls -1 {user_contrib}").splitlines()
        return subfiles

    def extract_source_files_from_folder(self, folder_name: str) -> List[Source]:
        """List `.v` files shipped with an installed package."""
        sources_path = os.path.join(self.opam_env_path, "lib/coq/user-contrib/", folder_name)
        subfiles = self.exec_cmd(f"find {sources_path}").splitlines()
        filepaths = [os.path.join(sources_path, file) for file in subfiles if file.endswith('.v')]
        return [self.get_source(filepath) for filepath in filepaths]

    def extract_opam_path(self, package_name: str, info_path: Dict[str, str]={}):
        """Resolve the OPAM installation path for a package."""
        opam_show = self.exec_cmd(f"opam show {package_name}")
        match = re.search(r'"logpath:([A-Za-z0-9_.-]+)"', opam_show)
        if not match:
            assert package_name in info_path and info_path[package_name], f'Missing info_path for {package_name}'
            return info_path[package_name], opam_show
        return match.group(1), opam_show

    def extract_source_files_from_package(self, package_name: str, info_path: Dict[str, str]={}) -> Dict[str, Any]:
        """List `.v` files shipped with an installed package."""
        fqn, opam_show = self.extract_opam_path(package_name, info_path)
        user_contrib_path = os.path.join(self.opam_env_path, "lib/coq/user-contrib/")
        sources_path = os.path.join(user_contrib_path, fqn.replace('.', '/'))
        subfiles = self.exec_cmd(f"find {sources_path}").splitlines()
        return {"fqn": fqn, "package_name": package_name, "root": user_contrib_path, "subfiles": [file for file in subfiles if file.endswith('.v')], "opam_show": opam_show}

    def get_source(self, filepath: str) -> Source:
        """Return a `Source` dataclass for a file inside the container."""
        content = self.read_file(filepath)
        return Source(path=filepath, content=content)
    
    def upload_source(self, content: str) -> Source:
        """Create a tmp file with content `content` inside the docker and return the correspondince Source object."""
        tmp_path = tempfile.NamedTemporaryFile().name + '.v'
        self.write_file(tmp_path, content, create_dir=False)
        return Source(tmp_path, content)