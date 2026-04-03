from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SemanticDocSearchClient:
    base_url: str
    route: str = "/search"
    api_key: str | None = None
    timeout: float = 30.0

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        url = self.base_url.rstrip("/") + "/" + self.route.lstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            url,
            json={"query": query, "k": int(k)},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            results = payload.get("results", [])
        elif isinstance(payload, list):
            results = payload
        else:
            raise ValueError(f"Invalid semantic search response: {type(payload).__name__}")
        if not isinstance(results, list):
            raise ValueError("Invalid semantic search response: `results` must be a list.")
        out: list[dict[str, Any]] = []
        for item in results:
            if isinstance(item, dict):
                out.append(item)
        return out[: max(0, int(k))]
