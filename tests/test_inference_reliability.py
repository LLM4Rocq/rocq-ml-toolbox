from __future__ import annotations

import importlib
import json
import time
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException


class FakePubSub:
    def __init__(self, messages: list[dict[str, Any]] | None = None):
        self._messages = list(messages or [])
        self.subscriptions: list[str] = []
        self.closed = False

    def subscribe(self, channel: str) -> None:
        self.subscriptions.append(channel)

    def unsubscribe(self, channel: str) -> None:
        if channel in self.subscriptions:
            self.subscriptions.remove(channel)

    def get_message(self, timeout: float = 0.0) -> dict[str, Any] | None:
        if self._messages:
            return self._messages.pop(0)
        return None

    def close(self) -> None:
        self.closed = True


class FakeRedis:
    def __init__(self):
        self.store: dict[str, Any] = {}
        self.published: list[tuple[str, str]] = []
        self._pubsub: FakePubSub | None = None
        self.lock_kwargs: dict[str, Any] | None = None

    def set(self, key: str, value: Any, ex: int | None = None) -> None:
        del ex
        self.store[key] = value

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def incr(self, key: str) -> int:
        value = int(self.store.get(key, 0)) + 1
        self.store[key] = value
        return value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def scan_iter(self, pattern: str):
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            for key in list(self.store):
                if key.startswith(prefix):
                    yield key
        elif pattern in self.store:
            yield pattern

    def exists(self, key: str) -> int:
        return int(key in self.store)

    def publish(self, channel: str, payload: str) -> None:
        self.published.append((channel, payload))

    def pubsub(self, ignore_subscribe_messages: bool = True) -> FakePubSub:
        del ignore_subscribe_messages
        if self._pubsub is None:
            self._pubsub = FakePubSub([])
        return self._pubsub

    def lock(self, key: str, timeout: int, blocking: bool, blocking_timeout: float | None):
        self.lock_kwargs = {
            "key": key,
            "timeout": timeout,
            "blocking": blocking,
            "blocking_timeout": blocking_timeout,
        }
        return FakeLock()

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis):
        self.redis = redis
        self._ops: list[tuple[str, Any]] = []

    def delete(self, key: str):
        self._ops.append(("delete", key))
        return self

    def execute(self):
        for op, key in self._ops:
            if op == "delete":
                self.redis.delete(key)
        self._ops.clear()
        return []


class FakeLock:
    def __init__(self):
        self.released = False

    def acquire(self):
        return True

    def release(self):
        self.released = True

    def extend(self, ttl: float, replace_ttl: bool):
        del ttl, replace_ttl


class FakePopen:
    _next_pid = 1000

    def __init__(self, argv, start_new_session=False):
        self.argv = argv
        self.kwargs = {"start_new_session": start_new_session}
        self.pid = FakePopen._next_pid
        FakePopen._next_pid += 1
        self._returncode = None

    def poll(self):
        return self._returncode

    def wait(self, timeout: float | None = None):
        del timeout
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self):
        self._returncode = 0

    def kill(self):
        self._returncode = -9


def _load_arbiter(monkeypatch):
    monkeypatch.setenv("NUM_PET_SERVER", "2")
    monkeypatch.setenv("PET_SERVER_START_PORT", "8765")
    monkeypatch.setenv("MAX_RAM_PER_PET", "0")
    monkeypatch.setenv("REDIS_URL", "redis://unused")
    monkeypatch.setenv("PET_CMD", "pet-server")
    import rocq_ml_toolbox.inference.arbiter as arbiter

    return importlib.reload(arbiter)


def test_arbiter_start_restart_uses_process_groups_and_generation(monkeypatch):
    arbiter = _load_arbiter(monkeypatch)
    fake_redis = FakeRedis()
    monkeypatch.setattr(arbiter, "redis_client", fake_redis)

    popen_calls: list[FakePopen] = []

    def fake_popen(argv, start_new_session=False):
        proc = FakePopen(argv, start_new_session=start_new_session)
        popen_calls.append(proc)
        return proc

    monkeypatch.setattr(arbiter.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(arbiter, "wait_until_pet_ready", lambda pet_idx, timeout_s=30.0: True)
    monkeypatch.setattr(arbiter.os, "killpg", lambda pid, sig: None)

    arbiter.start_pet_servers()
    assert len(popen_calls) == 2
    assert all(call.kwargs["start_new_session"] is True for call in popen_calls)
    assert fake_redis.get("generation:0") == 0
    assert fake_redis.get("generation:1") == 0
    assert fake_redis.get("pet_status:0") == arbiter.PetStatus.OK
    assert fake_redis.get("pet_status:1") == arbiter.PetStatus.OK

    ok = arbiter.restart_single_pet_server(0)
    assert ok is True
    assert len(popen_calls) == 3
    assert popen_calls[-1].kwargs["start_new_session"] is True
    assert fake_redis.get("generation:0") == 1
    assert fake_redis.get("pet_status:0") == arbiter.PetStatus.OK


def test_arbiter_defers_restart_while_pet_lock_is_held(monkeypatch):
    arbiter = _load_arbiter(monkeypatch)
    fake_redis = FakeRedis()
    monkeypatch.setattr(arbiter, "redis_client", fake_redis)

    fake_redis.set("pet_status:0", arbiter.PetStatus.RESTART_NEEDED)
    fake_redis.set("pet_lock:0", "LOCKED")

    restart_calls: list[int] = []
    monkeypatch.setattr(arbiter, "restart_single_pet_server", lambda pet_idx: restart_calls.append(pet_idx) or True)

    assert arbiter._maybe_restart_pet_server(0) is False
    assert restart_calls == []

    fake_redis.delete("pet_lock:0")
    assert arbiter._maybe_restart_pet_server(0) is True
    assert restart_calls == [0]


def test_session_manager_ensure_pet_ok_checks_status(monkeypatch):
    from rocq_ml_toolbox.inference import sessions

    req_uuid = uuid.UUID("00000000-0000-0000-0000-000000000123")
    monkeypatch.setattr(sessions.uuid, "uuid4", lambda: req_uuid)

    fake_redis = FakeRedis()
    req_id = str(req_uuid)
    fake_redis._pubsub = FakePubSub(
        [
            {
                "type": "message",
                "data": json.dumps({"id": req_id, "status": "RESTARTING"}),
            }
        ]
    )

    monkeypatch.setattr(sessions.redis.Redis, "from_url", lambda _: fake_redis)
    sm = sessions.SessionManager("redis://unused", num_pet_server=1)

    with pytest.raises(sessions.SessionManagerError) as exc:
        sm.ensure_pet_ok(0, timeout=1)

    assert exc.value.require_restart is True
    assert "status=RESTARTING" in str(exc.value)


def test_acquire_pet_lock_uses_unbounded_blocking_wait(monkeypatch):
    from rocq_ml_toolbox.inference import sessions

    fake_redis = FakeRedis()
    monkeypatch.setattr(sessions.redis.Redis, "from_url", lambda _: fake_redis)
    sm = sessions.SessionManager("redis://unused", num_pet_server=1)

    sm.acquire_pet_lock(0, timeout=42)

    assert fake_redis.lock_kwargs is not None
    assert fake_redis.lock_kwargs["blocking"] is True
    assert fake_redis.lock_kwargs["blocking_timeout"] is None
    assert fake_redis.lock_kwargs["timeout"] == 42


def test_server_health_endpoint_reflects_session_manager_snapshot():
    from rocq_ml_toolbox.inference import server

    unhealthy_req = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                sm=SimpleNamespace(health_snapshot=lambda: {"ok": False, "workers": {}})
            )
        )
    )

    with pytest.raises(HTTPException) as exc:
        server.health(unhealthy_req)
    assert exc.value.status_code == 503

    healthy = {"ok": True, "workers": {"0": {"status": "OK"}}}
    healthy_req = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                sm=SimpleNamespace(health_snapshot=lambda: healthy)
            )
        )
    )
    assert server.health(healthy_req) == healthy


def test_session_manager_strips_feedback_from_cached_state_by_default(monkeypatch):
    from rocq_ml_toolbox.inference import sessions
    from pytanque.protocol import State, RunParams

    fake_redis = FakeRedis()
    monkeypatch.setattr(sessions.redis.Redis, "from_url", lambda _: fake_redis)

    sm = sessions.SessionManager("redis://unused", num_pet_server=1)
    state = State(st=42, proof_finished=False, feedback=[(0, "some feedback")], generation=0)
    cleaned_state = sm._strip_feedback_from_state(state)
    assert state.feedback == [(0, "some feedback")]
    assert cleaned_state.feedback == []

    params = RunParams(st=state, tac="idtac.")
    cleaned_params = sm._strip_feedback_from_params(params)
    assert params.st.feedback == [(0, "some feedback")]
    assert cleaned_params.st.feedback == []

    sm_keep = sessions.SessionManager("redis://unused", num_pet_server=1, cache_feedback=True)
    kept_state = sm_keep._strip_feedback_from_state(state)
    kept_params = sm_keep._strip_feedback_from_params(params)
    assert kept_state.feedback == state.feedback
    assert kept_params.st.feedback == state.feedback


def test_session_manager_evicts_expired_session_and_related_keys(monkeypatch):
    from rocq_ml_toolbox.inference import sessions
    from rocq_ml_toolbox.inference.redis_keys import (
        mapping_state_key,
        mapping_tree_key,
        params_tree_key,
        session_key,
    )

    fake_redis = FakeRedis()
    monkeypatch.setattr(sessions.redis.Redis, "from_url", lambda _: fake_redis)

    sm = sessions.SessionManager(
        "redis://unused",
        num_pet_server=1,
        session_ttl_s=60,
        session_cleanup_interval_s=1,
    )
    session_id = sm.create_session()
    tree_id = "tree-id-a"
    fake_redis.set(
        mapping_tree_key(session_id),
        json.dumps({"mapping": {"0:1": tree_id}}),
    )
    fake_redis.set(params_tree_key(session_id, tree_id), json.dumps({"children": []}))

    raw_session = json.loads(fake_redis.get(session_key(session_id)))
    raw_session["updated_at"] = time.time() - 120
    fake_redis.set(session_key(session_id), json.dumps(raw_session))

    sm._next_session_cleanup_at = 0.0
    sm._maybe_evict_expired_sessions()

    assert fake_redis.get(session_key(session_id)) is None
    assert fake_redis.get(mapping_state_key(session_id)) is None
    assert fake_redis.get(mapping_tree_key(session_id)) is None
    assert fake_redis.get(params_tree_key(session_id, tree_id)) is None
    assert session_id not in sm.sessions_cache
    assert session_id not in sm.mappings_state_cache
    assert session_id not in sm.mappings_tree_cache
    assert session_id not in sm.params_trees_cache


def test_redis_cleanup_patterns_include_mapping_and_params_tree_keys():
    from rocq_ml_toolbox.inference.redis_keys import (
        ALL_KEYS_STAR,
        mapping_tree_key,
        params_tree_key,
    )

    assert mapping_tree_key("*") in ALL_KEYS_STAR
    assert params_tree_key("*", "*") in ALL_KEYS_STAR
