"""Configuration objects describing OPAM-based Coq environments."""

from dataclasses import dataclass, field
from typing import Dict

import yaml

@dataclass
class OpamConfig:
    """Configuration for building and querying a library Docker image."""

    name: str
    output: str
    tag: str
    packages: list[str]
    base_image: str
    opam_env_path: str
    user: str
    info_path: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "OpamConfig":
        """Load an `OpamConfig` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
