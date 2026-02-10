from pathlib import Path
from typing import Optional, Union
import re
import shlex
import sys
from abc import abstractmethod, ABC
import io
import tarfile
from pathlib import PurePosixPath

import docker
from docker.models.containers import Container

from .config import OpamConfig, DockerConfig
from ..parser.parser import Source

class BaseDocker(ABC):
    """Wraps Docker interactions."""

    def __init__(self, config:DockerConfig, kill_clone=False, rebuild=False, timeout_install=3600):
        """Start or reuse a container built from the given OPAM configuration."""
        super().__init__()
        self.client = docker.from_env()
        self.config = config
        image_name = config.name + ':' + config.tag
        
        if kill_clone: self._kill_clone(image_name)
        self._load_container(rebuild=rebuild, timeout_install=timeout_install)
        self.container: Container

    @staticmethod
    def _timeout_handler(signum, frame):
        """Signal handler for timeouts."""
        raise TimeoutError("Operation timed out")

    def _kill_clone(self, image_name):
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
    
    def _load_container(self, rebuild=False, **kwargs):
        image_name = self.config.name + ':' + self.config.tag
        user = self.config.user
        filterred_images = self.client.images.list(filters={'reference': image_name})
        if not filterred_images or rebuild:
            self._build_image(**kwargs)
        
        self.container = self.client.containers.run(
            image_name,
            detach=True,
            tty=False,
            stdin_open=False,
            user=user,
            command=["sleep", "infinity"],
            network_mode="host",
        )

    @abstractmethod
    def _build_image(self, timeout_install=3600):
        """Create a container image for the requested packages if needed."""
        pass

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

    def read_file(self, filepath: Union[str, Path], max_bytes=None, encoding="utf-8") -> str:
        """Read a file from the container filesystem."""
        filepath = str(filepath)
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

    def kill_container(self, container: Container, timeout=30):
        try:
            container.kill()
            container.wait(timeout=timeout)
        finally:
            container.remove(force=True)

    def close(self, timeout=30):
        """Stop and remove the underlying containers."""
        self.kill_container(self.container, timeout=timeout)

    def write_file(self, path, content: str, create_dir: bool = False, *, encoding: str = "utf-8") -> None:
        """
        Create/overwrite `path` inside the container with `content`.
        If create_dir=True, missing parent directories are created.
        """
        self._ensure_running()

        p = PurePosixPath(str(path))
        if not p.name:
            raise ValueError(f"Invalid file path: {path!r}")

        parent = str(p.parent) if str(p.parent) else "/"
        name = p.name

        if create_dir:
            api = self.client.api
            cmd = f"sh -lc 'mkdir -p -- {shlex.quote(parent)}'"
            exec_id = api.exec_create(self.container.id, cmd, stdout=True, stderr=True, tty=False)["Id"]
            _ = api.exec_start(exec_id)  # consume output
            code = api.exec_inspect(exec_id).get("ExitCode", 1)
            if code != 0:
                raise RuntimeError(f"Failed to create directory {parent!r} (exit code {code})")

        data = content.encode(encoding)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        buf.seek(0)

        ok = self.container.put_archive(parent, buf.getvalue())
        if not ok:
            raise RuntimeError(f"Failed to write file to {str(p)!r}")