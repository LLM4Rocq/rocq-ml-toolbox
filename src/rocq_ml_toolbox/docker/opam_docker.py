"""Helpers to manage Docker containers for OPAM-based Coq environments."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re
import time
import signal
import tempfile
import requests

from .config import OpamConfig
from .docker import BaseDocker
from ..parser.parser import Source
from .matches import match_paths
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
        # if self.config.user == "coq":
        #     cmd = f"""
        #     /home/{self.config.user}/miniconda/bin/conda run -n rocq-ml pip install pyyaml
        #     """
        #     self.exec_cmd(["bash", "-lc", cmd])
        #     self.container.commit(self.config.name, self.config.tag)
        if update_rocq_ml:
            self.exec_cmd([
                "bash",
                "-lc",
                "cd ~/rocq-ml-toolbox && git pull && git switch unified_fastapi"
            ])
            self.exec_cmd([
                "bash",
                "-lc",
                "cd ~/pytanque-repo && git pull && git switch pytanque_http"
            ])
            # cmd = f"""
            # /home/{self.config.user}/miniconda/bin/conda run -n rocq-ml pip install -U uvicorn fastapi
            # """
            # self.exec_cmd(["bash", "-lc", cmd])
            # self.container.commit(self.config.name, self.config.tag)


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
            network_mode="host"
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

    def start_inference_server(self, port=5000, timeout=600, workers=9, num_pet_server=4, pet_server_start_port=8765, max_ram_per_pet=4096):
        """Launch pet-server inside the container."""
        self.pet_port = port
        pet_server_path = os.path.join(self.config.opam_env_path, 'bin/pet-server')
        cmd = f"""
        /home/{self.config.user}/miniconda/bin/conda run -n rocq-ml rocq-ml-server -p {port} -w {workers} -t {timeout} -d -l \
        --num-pet-server {num_pet_server} \
        --pet-server-start-port {pet_server_start_port} \
        --max-ram-per-pet {max_ram_per_pet} \
        --pet-server-cmd {pet_server_path}
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
            time.sleep(1)
        log = self.exec_cmd("sh -lc 'tail -n +200 /tmp/gunicorn-error.log || true'")
        raise RuntimeError(f"rocq-ml-server failed to start on port {port}.\n{log}")

    def list_opam_folder(self) -> List[str]:
        "Extract all opam folder"
        user_contrib = os.path.join(self.opam_env_path, "lib/coq/user-contrib/")
        subfiles = self.exec_cmd(f"ls -1 {user_contrib}").splitlines()
        return subfiles

    def extract_files_from_target(self, target: str) -> List[str]:
        """List all files shipped with an installed package."""
        sources_path = os.path.join(self.opam_env_path, "lib/coq/user-contrib/", target)
        subfiles = self.exec_cmd(f"find {sources_path}").splitlines()
        filepaths = [os.path.join(sources_path, file) for file in subfiles]
        return filepaths

    def extract_source_files_from_corelib(self) -> List[Source]:
        sources_path = os.path.join(self.opam_env_path, "lib/coq/theories")
        subfiles = self.exec_cmd(f"find {sources_path}").splitlines()
        filepaths = [os.path.join(sources_path, file) for file in subfiles if file.endswith('.v')]
        return [self.get_source(filepath) for filepath in filepaths]

    def extract_source_files_from_target(self, target: str) -> List[Source]:
        """Extract source files shipped with an installed package."""
        subpaths = self.extract_files_from_target(target)
        filepaths = [file for file in subpaths if file.endswith('.v')]
        return [self.get_source(filepath) for filepath in filepaths]

    def extract_files_from_package(self, package: str) -> List[str]:
        """List all files shipped with an installed package."""
        package_path = os.path.join(self.opam_env_path, ".opam-switch/sources/", package)
        subfiles = self.exec_cmd(f"find {package_path}").splitlines()
        filepaths = [os.path.join(package_path, file) for file in subfiles]
        return filepaths

    def _map_vo_v_package_target(self, package: str, target: str) -> Tuple[List[Path], List[Path], List[Path]]:
        skeleton_vo = self.extract_files_from_target(target)
        skeleton_v = self.extract_files_from_package(package)

        skeleton_vo_target = [Path(p) for p in skeleton_vo if p.endswith('.vo')]
        skeleton_v_target = [Path(p) for p in skeleton_vo if p.endswith('.v')]
        skeleton_v_package = [Path(p) for p in skeleton_v if p.endswith('.v')]
        return skeleton_vo_target, skeleton_v_target, skeleton_v_package
    
    def copy_coq_files_from_package_to_target(self, package: str, target: str) -> List[str]:
        sources_path = os.path.join(self.opam_env_path, "lib/coq/user-contrib/", target)
        filepaths = self.extract_files_from_package(package)
        result = []
        for filepath in filepaths:
            filepath_p = Path(filepath)
            if filepath_p.name.startswith('_CoqProject'):
                coq_proj = filepath_p.name
                content = self.read_file(filepath)

                path_target = os.path.join(sources_path, coq_proj)
                self.write_file(path_target, content)
                result.append(filepath)
        
        vo_paths_target, v_paths_target, v_paths_package = self._map_vo_v_package_target(package, target)
        mapping, _ = match_paths(vo_paths_target, v_paths_package) 
        for package_p, target_p in mapping.items():
            package_target = package_p.with_suffix('.v')
            if package_target not in v_paths_target:
                content = self.read_file(target_p)
                self.write_file(package_target, content)
        return result

    def get_source(self, filepath: str) -> Source:
        """Return a `Source` dataclass for a file inside the container."""
        content = self.read_file(filepath)
        return Source(path=filepath, content=content)
    
    def upload_source(self, content: str) -> Source:
        """Create a tmp file with content `content` inside the docker and return the correspondince Source object."""
        tmp_path = tempfile.NamedTemporaryFile().name + '.v'
        self.write_file(tmp_path, content, create_dir=False)
        return Source(tmp_path, content)