from __future__ import annotations
import time
from typing import List, Optional, Dict, Tuple, Any, Iterator, cast
from contextlib import contextmanager
from functools import singledispatchmethod
from dataclasses import fields
import json
import uuid


from pytanque import Pytanque, PetanqueError
from pytanque.client import Params, Responses
from pytanque.response import BaseResponse, SessionResponse
from pytanque.params import SessionParams, PrimitiveParams
from pytanque.routes import RouteName

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
from .session_model import TacticsParent, TacticsTree, MappingState, Session, StateExtended, QueryKwargs

def require_session_response(res: BaseResponse, *, params: Params, route: RouteName) -> SessionResponse:
    if not isinstance(res, SessionResponse):
        raise SessionManagerError(
            f"{type(params).__name__} on {route} expects SessionResponse, got {type(res).__name__}"
        )
    return cast(SessionResponse, res)

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
    
    def mapping_state_cache_update(self, state: StateExtended, session: Session):
        """
        Update mapping_state_cache if the state is both outdated and not in it.
        """
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return
        mapping_state = self.mappings_state_cache[session.id]
        if state in mapping_state:
            state_map = mapping_state[state]
            if state_map.generation == current_generation:
                return
        # Since it didn't work, let's update MappingState cache
        self.mappings_state_cache[session.id] = MappingState.from_redis(session, self.redis_client)
    
    def tactics_tree_cache_update(self, state: StateExtended, session: Session):
        """
        Update tactics_trees_cache if the state is both outdated and not in it.
        """
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return
        tactics_tree = self.tactics_trees_cache[session.id]
        try:
            tactics_tree.find_path(state)
        except Exception:
            # state not found, let's update it.
            self.tactics_trees_cache[session.id] = TacticsParent.from_redis(session, self.redis_client)


    def update_state(self, state: StateExtended, session: Session, lock:Lock, timeout_run=60, timeout_get_state=120) -> StateExtended:
        """If state.generation is outdated, replay the tactics to recreate cache states on current pet-server."""
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return session # No need to replay
        worker = self._get_worker(session.pet_idx)

        self.mapping_state_cache_update(state, session)
        mapping_state = self.mappings_state_cache[session.id]
        if state in mapping_state:
            state_map = mapping_state[state]
            if state_map.generation == current_generation:
                return state_map
        
        self.tactics_tree_cache_update(state, session)
        tactics_tree = self.tactics_trees_cache[session.id]
        if tactics_tree:
            lock.extend(timeout_get_state+self.timeout_eps, replace_ttl=True)
            replay_session = tactics_tree.find_path(state)
            state_ext = mapping_state.get(tactics_tree.state_key, None)

            # if state_ext is outdated or None then regenerate it
            if not state_ext or state_ext.generation < current_generation:
                query_res = worker.query(**vars(tactics_tree.query_kwargs))
                res = worker.extract_response(tactics_tree.query_kwargs.route_name, query_res)
                if not isinstance(res, SessionResponse):
                    raise SessionManagerError(f"Tactics tree is assocaited to {tactics_tree.query_kwargs}, which doesn't corresponds to a `SessionResponse`")
                state = res.extract_response()
                state_ext = StateExtended.from_state(state, current_generation)
                mapping_state.add(state_ext)
            
            for node in replay_session:
                lock.extend(timeout_run+self.timeout_eps, replace_ttl=True)
                state_ext = mapping_state.get(node.state_key, None)
                # if state_ext is outdated or None then regenerate it
                if not state_ext or state_ext.generation < current_generation:
                    state = worker.run(state, node.tactic, timeout=timeout_run)
                    state_ext = StateExtended.from_state(state, current_generation)
                    mapping_state.add(state_ext)
            mapping_state.to_redis(session,self.redis_client)
        else:
            raise SessionManagerError(f"Session {session.id} doesn't have any TacticsParent")
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
        session = Session.from_redis(session_id, self.redis_client)
        pet_idx = session.pet_idx
        lock: Optional[Lock] = None
        try:
            lock = self.acquire_pet_lock(pet_idx, timeout=self.timeout_ok + self.timeout_eps)
            self.ensure_pet_ok(pet_idx, timeout=self.timeout_ok)
            # in rare cases pet server may have crashed between the first `from_redis`, and the Lock acquire
            session = Session.from_redis(session_id, self.redis_client)
            worker = self._get_worker(session.pet_idx)
            
            for field in fields(params):
                state = getattr(params, field.name)
                if isinstance(state, StateExtended):
                    new_state = self.update_state(state, session)
                    setattr(params, field.name, new_state)

            yield session, worker, lock, params

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
    
    @singledispatchmethod
    def _after_pet_call(
        self,
        params: Params,
        *,
        session: Session,
        route: RouteName,
        res: BaseResponse,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        # default: nothing to record
        return

    @_after_pet_call.register
    def _(
        self,
        params: SessionParams,
        *,
        session: Session,
        route: RouteName,
        res: BaseResponse,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        sres = require_session_response(res, params=params, route=route)
        state_ext = StateExtended.from_state(sres.extract_state(), gen)

        parent_state, tac = params.extract_parent()
        parent_ext = (
            parent_state
            if isinstance(parent_state, StateExtended)
            else StateExtended.from_state(parent_state, gen)
        )

        TacticsTree(state_ext.key, tac, parent_ext)

    @_after_pet_call.register
    def _(
        self,
        params: PrimitiveParams,
        *,
        session: Session,
        route: RouteName,
        res: BaseResponse,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        sres = require_session_response(res, params=params, route=route)
        state_ext = StateExtended.from_state(sres.extract_state(), gen)

        query_args = QueryKwargs(route, params, timeout=timeout)
        TacticsParent(state_ext.key, query_args)
    
    def _pet_call(
        self,
        session_id: str,
        route_name: RouteName,
        params: Params,
        timeout: Optional[float] = None,
    ) -> Any:
        with self._pet_ctx(session_id, params=params) as (session, worker, lock, updated_params):
            ttl = (timeout or self.timeout_ok) + self.timeout_eps
            lock.extend(ttl, replace_ttl=True)

            query_res = worker.query(updated_params, route_name, timeout=timeout)
            res = query_res.extract_response()

            gen = self.get_generation(session.pet_idx)
            self._after_pet_call(
                updated_params,
                session=session,
                route=route_name,
                res=res,
                gen=gen,
                timeout=timeout,
            )

            return res