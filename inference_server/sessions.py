from dataclasses import dataclass, asdict
import time
from typing import List, Optional, Dict, Tuple, Any
import os
import json
import uuid
import signal
import random

from pytanque import Pytanque, State, Goal, PetanqueError

import redis
from redis.lock import Lock

from inference_server.redis_keys import (
    PetStatus,
    session_key,
    pet_status_key,
    generation_key,
    pet_lock_key,
    monitor_epoch_key,
    session_lock_key,
    archived_sessions_key,
    session_assigned_idx_key,
    cache_state_key
)

@dataclass
class CacheState:
    pet_idx: int
    generation: int
    state: State

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "CacheState":
        return cls(
            pet_idx=raw['pet_idx'],
            generation=raw['generation'],
            state=State.from_json(raw['state'])
        )

    def to_json(self) -> dict:
        return {
            "pet_idx": self.pet_idx,
            "generation": self.generation,
            "state": self.state.to_json()
        }

@dataclass
class Session:
    session_id: str
    pet_idx: int                       # which pet-server index (0..num_pet_server-1)
    filepath: Optional[str]
    line: Optional[int]
    character: Optional[int]
    tactics: List[Tuple[State, str]]
    generation: Optional[int]          # pet-server generation where cached state is valid
    mapping_state: Dict[int, State]  # to map old states with new states in case of replay

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Session":
        tactics = [(State.from_json(st_json), tac) for st_json, tac in raw.get("tactics", [])]
        mapping_state_raw = raw.get("mapping_state", {})
        mapping_state = {int(k): State.from_json(v_json) for k, v_json in mapping_state_raw.items()}

        return cls(
            session_id=raw["session_id"],
            pet_idx=raw["pet_idx"],
            filepath=raw.get("filepath"),
            line=raw.get("line"),
            character=raw.get("character"),
            tactics=tactics,
            generation=raw.get("generation"),
            mapping_state=mapping_state,
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "pet_idx": self.pet_idx,
            "filepath": self.filepath,
            "line": self.line,
            "character": self.character,
            "tactics": [(st.to_json(), tac) for st, tac in self.tactics],
            "generation": self.generation,
            "mapping_state": {str(k): v.to_json() for k, v in self.mapping_state.items()},
        }

class UnresponsiveError(Exception):
    pass

class SessionManager:

    def __init__(self, pet_server_start_port: int=8765, num_pet_server: int=8, timeout_start_thm: int=60, timeout_run: int=30, timeout_ok: int=15):
        self.redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        self.ports = [pet_server_start_port + k for k in range(num_pet_server)]
        self.pytanques: List[Optional[Pytanque]] = [None] * num_pet_server
        self.worker_generations: List[Optional[int]] = [None] * num_pet_server
        self.sessions: Dict[str, Session] = {} # session_id -> Session
        self.num_pet_server = num_pet_server
        self.timeout_start_thm = timeout_start_thm
        self.timeout_run = timeout_run
        self.timeout_ok = timeout_ok
        self.max_timeout = max(timeout_start_thm, timeout_run)

    @staticmethod
    def handler(signum, frame):
        """Signal handler for timeouts."""
        raise UnresponsiveError("Operation timed out")

    def save_session(self, sess: Session):
        """Save session to Redis."""
        self.redis_client.set(
            session_key(sess.session_id),
            json.dumps(Session.to_dict(sess))
        )

    def get_session(self, session_id: str) -> Session:
        """Load session from Redis by session ID."""
        data = self.redis_client.get(session_key(session_id))
        if not data:
            raise KeyError("Unknown session_id")
        raw = json.loads(data)
        return Session.from_dict(raw)

    def get_generation(self, pet_idx: int) -> int:
        data = self.redis_client.get(generation_key(pet_idx))
        if not data:
            raise KeyError("Unknown session_id")
        return int(data)

    def get_assigned_idx(self) -> int:
        """Load session from Redis by session ID."""
        data = self.redis_client.get(session_assigned_idx_key())
        if not data:
            return 0
        return int(data)

    def _get_state_at_pos(self, pet_idx: int, filepath: str, line: int, character: int) -> State:
        worker = self._get_worker(pet_idx)
        id_str = str({
            "filepath": filepath,
            "line": line,
            "character": character,
            "pet_idx": pet_idx
        })
        raw = self.redis_client.get(cache_state_key(id_str))
        generation = self.get_generation(pet_idx)
    
        state: Optional[State] = None
        if not raw:
            state = worker.get_state_at_pos(filepath, line, character)
        else:
            data = json.loads(raw)
            cache_state = CacheState.from_json(data)
            if cache_state.generation != generation:
                state = worker.get_state_at_pos(filepath, line, character)
            else:
                state = cache_state.state
        
        cache_state = {
            "pet_idx": pet_idx,
            "generation": generation,
            "state": state.to_json()
        }
        self.redis_client.set(cache_state_key(id_str), json.dumps(cache_state))
        return state
        

    def _get_worker(self, pet_idx: int) -> Pytanque:
        """Return a connected Pytanque client for pet_idx, recreating if generation changed."""
        current_gen = self.get_generation(pet_idx)
        worker = self.pytanques[pet_idx]
        if worker is not None and self.worker_generations[pet_idx] == current_gen:
            return worker
        if worker is not None:
            try:
                worker.close()
            except Exception:
                pass

        worker = Pytanque("127.0.0.1", self.ports[pet_idx])
        try:
            worker.connect()
        except PetanqueError as e:
            raise UnresponsiveError(f"pet_idx {pet_idx} not reachable ({e}).") from e

        self.pytanques[pet_idx] = worker
        self.worker_generations[pet_idx] = current_gen
        return worker

    def _restart_worker(self, pet_idx: int) -> Pytanque:
        """Remove worker at pet_idx"""
        if self.pytanques[pet_idx]:
            self.pytanques[pet_idx].close()
            self.pytanques[pet_idx] = None

    def archive_session(self, sess: Session):
        """Store session data in Redis for archival purposes."""
        self.redis_client.rpush(archived_sessions_key(), json.dumps(asdict(sess)))

    def pet_status(self) -> bool:
        """Check if all pet-servers are in OK state."""
        for pet_idx in range(self.num_pet_server):
            state = self.redis_client.get(pet_status_key(pet_idx))
            if state is None or state.decode() != PetStatus.OK:
                return False
        return True

    def ensure_pet_ok(self, pet_idx: int, timeout=15, poll_interval=0.1):
        """Ensure that the pet-server at pet_idx is in OK state."""
        epoch_key = monitor_epoch_key(pet_idx)

        # Record the monitor epoch at the time we start
        start_epoch_raw = self.redis_client.get(epoch_key)
        start_epoch = int(start_epoch_raw) if start_epoch_raw is not None else 0

        # wait for a potential failure detection from the arbiter
        state = self.redis_client.get(pet_status_key(pet_idx))
        if state is None:
            raise KeyError(f"pet_idx {pet_idx} has no status")
        
        t0 = time.time()
        while time.time() - t0 < timeout:
            state = self.redis_client.get(pet_status_key(pet_idx))
            epoch_raw = self.redis_client.get(epoch_key)
            epoch = int(epoch_raw) if epoch_raw is not None else 0

            if state is not None and state.decode() == PetStatus.OK and epoch > start_epoch:
                return
            time.sleep(poll_interval)
        raise RuntimeError(f"pet_idx {pet_idx} not available (state={state})")

    def acquire_pet_lock(self, pet_idx: int, timeout: int=10) -> Lock:
        """
        Acquire a Redis lock for a given pet_idx.
        Returns the lock object (already acquired) or raises on failure.
        """
        lock = self.redis_client.lock(
            pet_lock_key(pet_idx),
            timeout=timeout,
            blocking=True
        )
        acquired = lock.acquire()
        if not acquired:
            raise UnresponsiveError(f"pet_idx {pet_idx} is busy")
        return lock

    def create_session(self, timeout: int=10) -> Session:
        """Create a new session with load-balanced pet-server assignment."""
        lock = self.redis_client.lock(
            session_lock_key(),
            timeout=timeout,
            blocking=True
        )
        acquired = lock.acquire()
        if not acquired:
            raise UnresponsiveError(f"Timeout when creating session.")
        self.redis_client.incr(session_assigned_idx_key())
        assigned_idx = self.get_assigned_idx() % self.num_pet_server
        uid = str(uuid.uuid4())
        sess = Session(uid, assigned_idx, None, None, None, [], None, {})
        self.save_session(sess)
        self.sessions[uid] = sess

        lock.release()
        return sess
    
    def check_session(self, sess: Session):
        """If sess.generation is outdated, replay the tactics to recreate cache states on current pet-server generation."""
        current_generation = self.get_generation(sess.pet_idx)
        if sess.generation == current_generation:
            return  # No need to replay
        
        worker = self._get_worker(sess.pet_idx)
        state = None
        if sess.filepath:
            try:
                signal.signal(signal.SIGALRM, SessionManager.handler)
                signal.alarm(self.timeout_start_thm)
                state = self._get_state_at_pos(sess.pet_idx, sess.filepath, sess.line, sess.character)
                for old_state, tactic in sess.tactics:
                    signal.alarm(self.timeout_run)
                    state = worker.run(state, tactic, verbose=False, timeout=self.timeout_run)
                    sess.mapping_state[old_state.hash] = state
            finally:
                signal.alarm(0)
        sess.generation = current_generation
        self.save_session(sess)  # Update session with new generation
    
    def send_kill_signal(self, pet_idx: int):
        """Send a kill signal to the pet-server at pet_idx."""
        self.redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
        self._restart_worker(pet_idx)

    def _failure_simulation(self):
        raise UnresponsiveError("Forced failure (Debugging purpose).")

    def start_thm(self, session_id: str, filepath: str, line: int, character: int, failure: bool=False) -> Tuple[State, List[Goal]]:
        """Start a theorem proving session for the theorem at the given position."""
        sess = self.get_session(session_id)
        pet_idx = sess.pet_idx
        lock: Optional[Lock] = None
        try:
            lock = self.acquire_pet_lock(pet_idx, timeout=self.max_timeout + 5)
            self.ensure_pet_ok(pet_idx, timeout=self.timeout_ok)
            sess = self.get_session(session_id)
            pet_idx = sess.pet_idx
            sess.generation = self.get_generation(pet_idx)
            worker = self._get_worker(pet_idx)

            if failure:
                self._failure_simulation()
            signal.signal(signal.SIGALRM, SessionManager.handler)
            signal.alarm(self.timeout_start_thm)

            state = self._get_state_at_pos(pet_idx, filepath, line, character)
            goals = worker.goals(state)
            signal.alarm(0)  # Disable alarm

            if sess.tactics:
                # archive session if non empty previous session
                self.archive_session(sess)
            sess.filepath = filepath
            sess.line = line
            sess.character = character
            sess.tactics = [(state, "")]
            sess.mapping_state = {}

            self.save_session(sess)
            return state, goals
        except KeyError: raise
        except PetanqueError: raise
        except Exception as e:
            # No response, need to restart pet server
            self.send_kill_signal(pet_idx)
            raise UnresponsiveError(f"Start theorem failed ({e}); kill signal sent to pet-server.") from e
        finally:
            signal.alarm(0)
            if lock is not None:
                try:
                    lock.release()
                except redis.exceptions.LockError:
                    # Lock might have expired; safe to ignore
                    pass

    def run(self, session_id: str, state: State, tactic: str, failure: bool=False) -> Tuple[State, List[Goal]]:
        """Execute a given tactic on the current proof state."""
        sess = self.get_session(session_id)
        pet_idx = sess.pet_idx
        lock: Optional[Lock] = None
        try:
            lock = self.acquire_pet_lock(pet_idx, timeout=self.max_timeout + 5)
            self.ensure_pet_ok(pet_idx, timeout=self.timeout_ok)
            self.check_session(sess)
            sess = self.get_session(session_id)
            if state.hash in sess.mapping_state:
                state = sess.mapping_state[state.hash] # update state, required in case of replay
            worker = self._get_worker(sess.pet_idx)
            
            if failure:
                self._failure_simulation()
            signal.signal(signal.SIGALRM, SessionManager.handler)
            signal.alarm(self.timeout_run)
            state = worker.run(state, tactic, verbose=False, timeout=self.timeout_run)
            goals = worker.goals(state)
            signal.alarm(0)

            sess.tactics.append((state, tactic))
            self.save_session(sess)

            return state, goals
        except KeyError: raise
        except PetanqueError: raise
        except Exception as e:
            # No response, need to restart pet server
            self.send_kill_signal(pet_idx)
            raise UnresponsiveError(
                f"Tactic execution failed ({e}); kill signal sent to pet-server."
            ) from e
        finally:
            signal.alarm(0)
            if lock is not None:
                try:
                    lock.release()
                except redis.exceptions.LockError:
                    # Lock might have expired; safe to ignore
                    pass