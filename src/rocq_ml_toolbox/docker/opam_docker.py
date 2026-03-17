"""Helpers to manage Docker containers for OPAM-based Coq environments."""

import os
import shlex
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

# Matches:
#   Declare ML Module "foo" "bar:baz.qux".
# across spaces/newlines, but only inside Declare ML Module commands.
_DECLARE_ML_MODULE_RE = re.compile(
    r'(\bDeclare\s+ML\s+Module\b)'      # command head
    r'(?P<mods>(?:\s*"[^"]*")+)'        # one or more quoted strings
    r'(\s*\.)',                         # terminating dot
    re.MULTILINE,
)

_QUOTED_RE = re.compile(r'"([^"]*)"')

# Heuristic for a modern/public plugin name, e.g. coq-core.plugins.ssreflect
_PUBLIC_RE = re.compile(r'^[A-Za-z0-9_+-]+(?:\.[A-Za-z0-9_+-]+)+$')


def _normalize_plugin_name(name: str) -> str:
    # Already modern syntax: keep unchanged
    if ":" not in name:
        return name

    legacy, public = name.split(":", 1)

    # Only rewrite obvious legacy:public declarations.
    # If it does not look like a public findlib name, leave it unchanged.
    if legacy and _PUBLIC_RE.fullmatch(public):
        return public

    return name


def normalize_declare_ml_module_syntax(source: str) -> str:
    def repl_decl(match: re.Match[str]) -> str:
        head = match.group(1)
        mods = match.group("mods")
        tail = match.group(3)

        def repl_quoted(qm: re.Match[str]) -> str:
            original = qm.group(1)
            rewritten = _normalize_plugin_name(original)
            return f'"{rewritten}"'

        new_mods = _QUOTED_RE.sub(repl_quoted, mods)
        return f"{head}{new_mods}{tail}"

    return _DECLARE_ML_MODULE_RE.sub(repl_decl, source)

class OpamDocker(BaseDocker):
    """Wraps Docker interactions for extracting data from an OPAM switch."""

    def __init__(self, config:OpamConfig, rebuild=False, kill_clone=True, update_rocq_ml=False):
        """Start or reuse a container built from the given OPAM configuration."""
        super().__init__(config, kill_clone=kill_clone, rebuild=rebuild)
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

    def wait_redis_ready(self, port=6379, timeout=10.0, interval=0.1):
        """Wait until Redis answers PING with PONG."""
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            try:
                out = self.exec_cmd(
                    ["redis-cli", "-p", str(port), "ping"]
                ).strip()
                if out == "PONG":
                    return
                last_error = f"unexpected response: {out!r}"
            except Exception as e:
                last_error = str(e)

            time.sleep(interval)

        raise TimeoutError(
            f"Redis was not ready within {timeout}s on port {port}. "
            f"Last error: {last_error}"
        )

    def start_redis_server(self, port=6379):
        """Launch redis-server inside the container."""
        self.exec_cmd_async(f"bash -lc redis-server --port {port}")
        self.wait_redis_ready(port)


    def start_inference_server(self, port=5000, redis_port=6379, timeout=600, workers=9, num_pet_server=4, pet_server_start_port=8765, max_ram_per_pet=4096):
        """Launch pet-server inside the container."""
        self.start_redis_server(redis_port)
        self.pet_port = pet_server_start_port
        opam_bin = os.path.join(self.config.opam_env_path, "bin")
        pet_server_path = os.path.join(opam_bin, "pet-server")
        conda_env_bin = f"/home/{self.config.user}/miniconda/envs/rocq-ml/bin"
        conda_root_bin = f"/home/{self.config.user}/miniconda/bin"
        rocq_ml_server_bin = f"/home/{self.config.user}/miniconda/envs/rocq-ml/bin/rocq-ml-server"
        redis_url = f"redis://localhost:{redis_port}/0"

        cmd = f"""\
        set -e
        eval "$(opam env)"
        export PATH={shlex.quote(opam_bin)}:{shlex.quote(conda_env_bin)}:{shlex.quote(conda_root_bin)}:$PATH
        {shlex.quote(rocq_ml_server_bin)} -p {port} -w {workers} -t {timeout} -d \\
          --redis-url {shlex.quote(redis_url)} \\
          --num-pet-server {num_pet_server} \\
          --pet-server-start-port {pet_server_start_port} \\
          --max-ram-per-pet {max_ram_per_pet} \\
          --pet-server-cmd {shlex.quote(pet_server_path)}
        """
        self.exec_cmd(["bash", "-lc", cmd], check=False)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        logs = []
        for log_path in (
            f"/home/{self.config.user}/arbiter.log",
            f"/home/{self.config.user}/server.log",
            "/tmp/gunicorn-error.log",
        ):
            out = self.exec_cmd(f"sh -lc 'tail -n 200 {shlex.quote(log_path)} 2>/dev/null || true'").strip()
            if out:
                logs.append(f"== {log_path} ==\n{out}")
        logs_txt = "\n\n".join(logs) if logs else "(no logs found)"
        raise RuntimeError(f"rocq-ml-server failed to start on port {port}.\n{logs_txt}")

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
        target_path = os.path.join(self.opam_env_path, "lib/coq/user-contrib/Corelib_duplicate")

        self.cp(sources_path, target_path)
        subfiles = self.exec_cmd(f"find {target_path}").splitlines()
        return [self.get_source(filepath) for filepath in subfiles if filepath.endswith('.v')]

    def extract_source_files_from_target(self, target: str) -> List[Source]:
        """Extract source files shipped with an installed package."""
        subpaths = self.extract_files_from_target(target)
        filepaths = [file for file in subpaths if file.endswith('.v')]
        return [self.get_source(filepath) for filepath in filepaths]

    def extract_target_name(self, package: str) -> Optional[str]:
        """Attempt to extract lib name from package."""
        package_path = os.path.join(self.opam_env_path, ".opam-switch/sources/", package)
        subfiles = self.exec_cmd(f"find {package_path}").splitlines()

        proj_filename = None
        if '_CoqProject' in subfiles:
            proj_filename = '_CoqProject'
        if '_RocqProject' in subfiles:
            proj_filename = '_RocqProject'
        
        if proj_filename:
            proj_filepath = os.path.join(package_path, proj_filename)
            content = self.read_file(proj_filepath)
            m = re.search(r'-R\s+\S+\s+(\S+)', content)
            if m:
                return m.group(1)
        return None

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
    
    def add_coqproject(self, target: str, extra_coq_proj_args: List[str]=[], target_replace: Optional[str]=None):
        sources_path = os.path.join(self.opam_env_path, "lib/coq/user-contrib/", target)
        path_target = os.path.join(sources_path, '_CoqProject')
        if target_replace:
            target = target_replace
        content = f'-R . {target}\n'
        if extra_coq_proj_args:
            content += '\n'.join(extra_coq_proj_args)
        self.write_file(path_target, content)
    
    def copy_coq_files_from_package_to_target(self, package: str, target: str):
        vo_paths_target, v_paths_target, v_paths_package = self._map_vo_v_package_target(package, target)
        mapping, _ = match_paths(vo_paths_target, v_paths_package) 
        for package_p, target_p in mapping.items():
            package_target = package_p.with_suffix('.v')
            if package_target not in v_paths_target:
                content = self.read_file(target_p)
                self.write_file(package_target, content)

    def copy_elpi_files_from_package_to_target(self, package: str, target: str):
        vo_paths_target, _, v_paths_package = self._map_vo_v_package_target(package, target)
        mapping, _ = match_paths(vo_paths_target, v_paths_package)

        # package dir -> installed dir
        dir_map = {src_v.parent: dst_vo.parent for dst_vo, src_v in mapping.items()}
        existing = {Path(p) for p in self.extract_files_from_target(target)}

        for src in map(Path, self.extract_files_from_package(package)):
            if src.suffix != ".elpi":
                continue
            for src_root, dst_root in sorted(dir_map.items(), key=lambda kv: len(kv[0].parts), reverse=True):
                try:
                    rel = src.relative_to(src_root)
                except ValueError:
                    continue
                dst = dst_root / rel
                if dst not in existing:
                    self.write_file(str(dst), self.read_file(str(src)))
                    existing.add(dst)
                break

    def remove_legacy_plugin_from_source(self, source: Source):
        new_content = normalize_declare_ml_module_syntax(source.content)
        source.content = new_content
        self.write_file(source.path, new_content)

    def get_source(self, filepath: str) -> Source:
        """Return a `Source` dataclass for a file inside the container."""
        content = self.read_file(filepath)
        return Source(path=filepath, content=content)
    
    def upload_source(self, content: str) -> Source:
        """Create a tmp file with content `content` inside the docker and return the correspondince Source object."""
        tmp_path = tempfile.NamedTemporaryFile().name + '.v'
        self.write_file(tmp_path, content, create_dir=False)
        return Source(tmp_path, content)
