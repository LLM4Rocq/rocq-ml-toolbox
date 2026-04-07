from __future__ import annotations

from typing import Any

import pytest
import requests

from src.rocq_ml_toolbox.inference.client import PytanqueExtended


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")

    def json(self) -> Any:
        return self._payload


def test_client_access_libraries_and_read_docstrings_validation(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any]):
        calls.append((url, json))
        if url.endswith("/access_libraries"):
            return _FakeResponse({"env": "coq-x", "nodes": [], "file_index": {}})
        if url.endswith("/read_docstrings"):
            return _FakeResponse({"docstrings": [{"uid": "u", "docstring": "d"}]})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "post", fake_post)
    client = PytanqueExtended("127.0.0.1", 5000)

    toc = client.access_libraries("coq-x")
    assert toc["env"] == "coq-x"
    docs = client.read_docstrings("/tmp/A.v")
    assert docs and docs[0]["uid"] == "u"

    assert calls[0][1]["env"] == "coq-x"
    assert calls[1][1]["source"] == "/tmp/A.v"


def test_client_access_libraries_without_env(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any]):
        calls.append((url, json))
        if url.endswith("/access_libraries"):
            return _FakeResponse({"env": "coq-auto", "nodes": [], "file_index": {}})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "post", fake_post)
    client = PytanqueExtended("127.0.0.1", 5000)

    toc = client.access_libraries()
    assert toc["env"] == "coq-auto"
    assert calls[0][1]["env"] is None


def test_client_read_write_file_validation(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any]):
        calls.append((url, json))
        if url.endswith("/read_file"):
            return _FakeResponse(
                {
                    "path": json["path"],
                    "content": "abc",
                    "offset": json["offset"],
                    "next_offset": 3,
                    "eof": True,
                    "total_chars": 3,
                }
            )
        if url.endswith("/write_file"):
            return _FakeResponse({"path": json["path"], "bytes_written": len(json["content"]), "size": 10})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(requests, "post", fake_post)
    client = PytanqueExtended("127.0.0.1", 5000)

    out = client.read_file("/tmp/a.v", offset=0, max_chars=10)
    assert out["content"] == "abc"
    out2 = client.read_file("theories/Demo.v", offset=0, max_chars=10, path_mode="coq_lib_relative")
    assert out2["content"] == "abc"
    wrote = client.write_file("/tmp/a.v", content="xyz", offset=0, truncate=True)
    assert wrote["bytes_written"] == 3
    first_read_payload = calls[0][1]
    second_read_payload = calls[1][1]
    assert "path_mode" not in first_read_payload
    assert second_read_payload["path_mode"] == "coq_lib_relative"


def test_client_raises_on_invalid_response(monkeypatch):
    def fake_post(url: str, json: dict[str, Any]):
        if url.endswith("/access_libraries"):
            return _FakeResponse({"env": "coq-x", "nodes": "bad", "file_index": {}})
        return _FakeResponse({})

    monkeypatch.setattr(requests, "post", fake_post)
    client = PytanqueExtended("127.0.0.1", 5000)
    with pytest.raises(ValueError):
        client.access_libraries("coq-x")
