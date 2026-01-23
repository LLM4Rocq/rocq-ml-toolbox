from __future__ import annotations
import time
from typing import List, Optional, Dict, Tuple, Any, Iterator
from contextlib import contextmanager
import json
import uuid

from pytanque import Pytanque, PetanqueError
from pytanque.client import Params, Responses
from .rpc_registry import rpc

import redis
from redis.lock import Lock

from .redis_keys import (
    PetStatus,
    session_key,
    pet_status_key,
    generation_key,
    pet_lock_key,
    monitor_epoch_key,
    archived_sessions_key,
    session_assigned_idx_key
)
from .session_model import TacticsParent, MappingState, Session, StateExtended

class SessionManagerError(Exception):
    def __init__(self, message, require_restart=False):
        self.message = message
        self.require_restart = require_restart


class SessionManager:

    def __init__(self, redis_url: str, pet_server_start_port: int=8765, num_pet_server: int=4, timeout_ok: int=15, timeout_eps: int=10):
        self.redis_client = redis.Redis.from_url(redis_url)
        self.ports = [pet_server_start_port + k for k in range(num_pet_server)]
        self.pytanques: List[Optional[Pytanque]] = [None] * num_pet_server
        self.worker_generations: List[Optional[int]] = [None] * num_pet_server
        self.sessions_cache: Dict[str, Session] = {} # session_id -> Session
        self.mappings_state_cache: Dict[str, MappingState] = {}
        self.tactics_trees_cache: Dict[str, Optional[TacticsParent]] = {}
        self.num_pet_server = num_pet_server
        self.timeout_ok = timeout_ok
        self.timeout_eps = timeout_eps

    def get_generation(self, pet_idx: int) -> int:
        data = self.redis_client.get(generation_key(pet_idx))
        if not data:
            raise SessionManagerError("Unknown session_id")
        return int(data)

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
        worker.connect()
        self.pytanques[pet_idx] = worker
        self.worker_generations[pet_idx] = current_gen
        return worker

    def _restart_worker(self, pet_idx: int) -> Pytanque:
        """Remove worker at pet_idx"""
        if self.pytanques[pet_idx]:
            self.pytanques[pet_idx].close()
            self.pytanques[pet_idx] = None

    def archive_session(self, session: Session, tactics_tree: TacticsParent):
        """Store session data in Redis for archival purposes."""
        entry = {
            "session": session.to_json(),
            "tactics_tree": tactics_tree.to_json()
        }
        self.redis_client.rpush(archived_sessions_key(), json.dumps(entry))

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
            raise SessionManagerError(f"pet_idx {pet_idx} has no status")
        
        t0 = time.time()
        while time.time() - t0 < timeout:
            state = self.redis_client.get(pet_status_key(pet_idx))
            epoch_raw = self.redis_client.get(epoch_key)
            epoch = int(epoch_raw) if epoch_raw is not None else 0

            if state is not None and state.decode() == PetStatus.OK and epoch > start_epoch:
                return
            time.sleep(poll_interval)
        raise SessionManagerError(f"pet_idx {pet_idx} not available (state={state})", require_restart=True)

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
            raise SessionManagerError(f"pet_idx {pet_idx} is busy")
        return lock

    def create_session(self) -> str:
        """Create a new session with load-balanced pet-server assignment."""
        assigned_idx = self.redis_client.incr(session_assigned_idx_key())
        assigned_idx = assigned_idx % self.num_pet_server
        uid = str(uuid.uuid4())
        session = Session(id=uid, pet_idx=assigned_idx)
        session.to_redis(self.redis_client)
        self.sessions_cache[session.id] = session
        return session.id
    
    def update_state(self, state: StateExtended, session: Session, lock:Lock, timeout_run=60, timeout_get_state=120) -> StateExtended:
        """If state.generation is outdated, replay the tactics to recreate cache states on current pet-server."""
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return session # No need to replay
        worker = self._get_worker(session.pet_idx)

        mapping_state = self.mappings_state_cache[session.id]
        if state in mapping_state:
            state_map = mapping_state[state]
            if state_map.generation == current_generation:
                return state_map
        
        tactics_tree = self.tactics_trees_cache[session.id]
        state_ext = None
        if tactics_tree:
            lock.extend(timeout_get_state+self.timeout_eps, replace_ttl=True)
            replay_session = tactics_tree.find_path(state)
            state = worker.query(**vars(tactics_tree.query_kwargs))
            for node in replay_session:
                lock.extend(timeout_run+self.timeout_eps, replace_ttl=True)
                state = worker.run(state, node.tactic, timeout=timeout_run)
                mapping_state.add(state_ext)
        else:
            raise SessionManagerError(f"Session {session.id} doesn't have any TacticsParent")
        state_ext = StateExtended.from_state(state, current_generation)
        return state_ext
    
    def send_kill_signal(self, pet_idx: int):
        """Send a kill signal to the pet-server at pet_idx."""
        self.redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
        self._restart_worker(pet_idx)

    @contextmanager
    def _pet_ctx(
        self,
        session_id: str,
        params: Params
    ) -> Iterator[Tuple[Session, Pytanque, Lock, Params]]:
        sess = self.get_session(session_id)
        pet_idx = sess.pet_idx
        lock: Optional[Lock] = None
        try:
            lock = self.acquire_pet_lock(pet_idx, timeout=self.timeout_ok + self.timeout_eps)
            self.ensure_pet_ok(pet_idx, timeout=self.timeout_ok)
            sess = self.get_session(session_id)

            worker = self._get_worker(sess.pet_idx)

            if state_ext:
                sess = self.check_session(sess, lock) # refresh after potential replay
                state_ext_str = state_ext.to_json_string()
                if state_ext_str in sess.mapping_state:
                    state_ext = sess.mapping_state[state_ext_str]

            yield sess, worker, lock, params

        except PetanqueError as e:
            # if petanque error is related to a timeout, then send kill signal to the underlying pet server.
            if e.code == -33_000:
                self.send_kill_signal(pet_idx)
            raise e
        except SessionManagerError as e:
            if e.require_restart:
                self.send_kill_signal(pet_idx)
            raise e
        except Exception as e:
            # if unknown issue then send kill signal to the underlying pet server.
            self.send_kill_signal(pet_idx)
            raise e
        finally:
            if lock is not None:
                try:
                    lock.release()
                except redis.exceptions.LockError:
                    pass

    def _pet_call(
        self,
        session_id: str,
        route: str,
        params: Params,
        res_cls: type[Responses],
        is_session: bool,
        parent: str='',
        cmd: str='',
        timeout: Optional[float]=None
    ) -> Any:
        with self._pet_ctx(
            session_id,
            params=params
        ) as (sess, worker, lock, updated_params):
            lock.extend(timeout, replace_ttl=True)
            try:
                parent = getattr(params, parent, None)
                cmd = getattr(params, cmd, '')
                res_raw = worker.query(updated_params, route, timeout=timeout)
                res = res_cls.extract_response(res_raw)
                if is_session:
                    sess.tactics.append(())
                return res
            finally:
                lock.release()