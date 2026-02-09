from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Self, List, Any, Union, Tuple
from pathlib import Path
import yaml

@dataclass
class DockerConfig:
    """Configuration for building a library Docker image."""
    name: str
    tag: str
    base_image: str
    user: str

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Self:
        """Load an `OpamConfig` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

@dataclass
class Target:
    lib: str
    packages: List[str]

    @classmethod
    def from_json(cls, x: dict) -> Target:
        return cls(**x)

    def to_json(self) -> dict:
        return {
            "lib": self.lib,
            "packages": self.packages
        }

    def is_inside(self, other_target: Target) -> bool:
        if self.lib != other_target.lib:
            return False
        for package in self.packages:
            if package not in other_target.packages:
                return False
        return True


@dataclass
class OpamConfig(DockerConfig):
    """Configuration for building an Opam Docker image."""
    opam_env_path: str
    packages: list[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    pins: list[str] = field(default_factory=list)
    targets: List[Target] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Self:
        """Load an `OpamConfig` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        data['targets'] = [Target.from_json(t) for t in data['targets']]
        return cls(**data)

def solve_deps_config(opam_configs: List[DockerConfig]) -> List[Tuple[DockerConfig, List[Target]]]:
    pass