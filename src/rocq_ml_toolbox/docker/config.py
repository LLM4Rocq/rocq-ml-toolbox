from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Self

import yaml

@dataclass
class DockerConfig:
    """Configuration for building a library Docker image."""
    name: str
    tag: str
    base_image: str
    user: str

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Load an `OpamConfig` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

@dataclass
class OpamConfig(DockerConfig):
    """Configuration for building an Opam Docker image."""
    opam_env_path: str
    opam_pins: list[str]
    opam_packages: list[str]
    info_path: Dict[str, str] = field(default_factory=dict)
