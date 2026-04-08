from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SemanticDocSearchClient:
    base_url: str
    route: str = "/search"
    api_key: str | None = None
    env: str = "coq-mathcomp"
    timeout: float = 30.0

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_result(cls, item: dict[str, Any]) -> dict[str, Any]:
        score = cls._to_float_or_none(item.get("score"))
        normalized = dict(item)
        normalized["score"] = score
        normalized["logical_path"] = item.get("logical_path")
        normalized["relative_path"] = item.get("relative_path")
        localization = item.get("localization")
        normalized["localization"] = localization if isinstance(localization, dict) else {}
        return normalized

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        url = self.base_url.rstrip("/") + "/" + self.route.lstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            url,
            json={"query": query, "env": self.env, "k": int(k)},
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
                out.append(self._normalize_result(item))
        out.sort(key=lambda r: (r.get("score") is not None, r.get("score")), reverse=True)
        return out[: max(0, int(k))]
