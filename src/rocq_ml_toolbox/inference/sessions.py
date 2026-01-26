from __future__ import annotations
import time
from typing import List, Optional, Dict, Tuple, Any, Iterator, cast
from contextlib import contextmanager
from functools import singledispatchmethod
from dataclasses import fields
import json
import uuid


from pytanque import Pytanque, PetanqueError
from pytanque.client import Params
from pytanque.response import BaseResponse, SessionResponse, Response
from pytanque.params import SessionParams, PrimitiveParams
from pytanque.routes import RouteName

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
from .session_model import TacticsTree, MappingState, Session, State, QueryKwargs

def require_session_response(res: BaseResponse, *, params: Params, route_name: RouteName, **kwargs) -> SessionResponse:
    if not isinstance(res, SessionResponse):
        raise SessionManagerError(
            f"{type(params).__name__} on {route_name} expects SessionResponse, got {type(res).__name__}"
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
        self.tactics_trees_cache: Dict[str, Optional[TacticsTree]] = {}
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

    def archive_session(self, session: Session, tactics_tree: TacticsTree):
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
        self.mappings_state_cache[session.id] = MappingState()
        self.tactics_trees_cache[session.id] = None

        self.mappings_state_cache[session.id].to_redis(session, self.redis_client)
        return session.id
    
    def mapping_state_cache_update(self, state: State, session: Session):
        """
        Update mapping_state_cache if the state is both outdated and not in it.
        """
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return
        # check if session is in mappings_state_cache
        if session.id not in self.mappings_state_cache:
            self.mappings_state_cache[session.id] = MappingState.from_redis(session, self.redis_client)
            return
        mapping_state = self.mappings_state_cache[session.id]

        if state not in mapping_state:
            self.mappings_state_cache[session.id] = MappingState.from_redis(session, self.redis_client)
    
    def tactics_tree_cache_update(self, state: State, session: Session):
        """
        Update tactics_trees_cache if the state is not in it.
        """
        # check if session is in tactics_tree_cache
        if session.id not in self.tactics_trees_cache or \
            not self.tactics_trees_cache[session.id]:
            self.tactics_trees_cache[session.id] = TacticsTree.from_redis(session, self.redis_client)
            return
        tactics_tree = self.tactics_trees_cache[session.id]
        try:
            tactics_tree.find_path(state)
        except Exception:
            # state not found, let's update the tactics_tree_cache.
            self.tactics_trees_cache[session.id] = TacticsTree.from_redis(session, self.redis_client)

    def update_state(self, state: State, session: Session, lock:Lock, timeout_run=60, timeout_get_state=120) -> State:
        """If state.generation is outdated, replay the tactics to recreate cache states on current pet-server."""
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return state # No need to replay
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
            print(f"START REPLAY for {state}")
            for node in replay_session:
                print("REPLAY")
                print(node.query_kwargs.params)
                state = mapping_state.get(node.state_key, None)
                    
                # if state is outdated or None then regenerate it
                if not state or state.generation < current_generation:
                    query_kwargs = node.query_kwargs.from_json(node.query_kwargs.to_json())
                    query_kwargs.params = self._update_params(query_kwargs.params, session, lock)
                    query_res = worker.query(**vars(query_kwargs))
                    query_res = require_session_response(query_res, **vars(query_kwargs))
                    state = query_res.extract_response()
                    state.generation = current_generation
                    mapping_state.add(node.state_key, state)
            print("END REPLAY")
            mapping_state.to_redis(session,self.redis_client)
        else:
            raise SessionManagerError(f"Session {session.id} doesn't have any TacticsParent")
        return state
    
    def send_kill_signal(self, pet_idx: int):
        """Send a kill signal to the pet-server at pet_idx."""
        self.redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
        self._restart_worker(pet_idx)

    def _update_params(
            self,
            params: Params,
            session: Session,
            lock: Lock
    ):
        new_params = params.from_json(params.to_json())
        for field in fields(new_params):
            state = getattr(new_params, field.name)
            if isinstance(state, State):
                new_state = self.update_state(state, session, lock)
                setattr(new_params, field.name, new_state)
        return new_params

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
            updated_params = self._update_params(params, session, lock)
            yield session, worker, lock, updated_params

        except PetanqueError as e:
            # if petanque error is related to a timeout, then send kill signal to the underlying pet server.
            # if e.code == -33_000:
            #     self.send_kill_signal(pet_idx)
            raise e
        except SessionManagerError as e:
            # if e.require_restart:
            #     self.send_kill_signal(pet_idx)
            raise e
        except Exception as e:
            # if unknown issue then send kill signal to the underlying pet server.
            # self.send_kill_signal(pet_idx)
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
        updated_params: Params,
        *,
        session: Session,
        route_name: RouteName,
        res: BaseResponse,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        # default: nothing to record
        return res.extract_response()

    @_after_pet_call.register
    def _(
        self,
        params: SessionParams,
        updated_params: SessionParams,
        *,
        session: Session,
        route_name: RouteName,
        res: BaseResponse,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        print(f"Session Params {params}")
        sres = require_session_response(res, params=params, route_name=route_name)
        state = sres.extract_response()
        state.generation = gen

        parent_state = params.extract_parent()
        self.tactics_tree_cache_update(parent_state, session)
        tactics_tree = self.tactics_trees_cache[session.id]

        parent_node = tactics_tree.find_node(parent_state)

        query_args = QueryKwargs(route_name, params, timeout=timeout)
        child = TacticsTree.from_state(state, query_args)
        parent_node.add_child(child)
        tactics_tree.to_redis(session, self.redis_client)
        print(f"End call")
        return state

    @_after_pet_call.register
    def _(
        self,
        params: PrimitiveParams,
        updated_params: PrimitiveParams,
        *,
        session: Session,
        route_name: RouteName,
        res: BaseResponse,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        print(f"Primitive Params {params}")
        sres = require_session_response(res, params=updated_params, route_name=route_name)
        state = sres.extract_response()
        state.generation = gen
        query_args = QueryKwargs(route_name, params, timeout=timeout)
        tactics_tree = TacticsTree.from_state(state, query_args)
        tactics_tree.to_redis(session, self.redis_client)
        print(f"End call")
        return state
    
    def _pet_call(
        self,
        request_id: int,
        session_id: str,
        route_name: RouteName,
        params: Params,
        timeout: Optional[float] = None,
    ) -> Any:
        # TODO: set_workspace is not manage right now
        print("NEW REQUEST")
        print(params)
        with self._pet_ctx(session_id, params=params) as (session, worker, lock, updated_params):
            ttl = (timeout or self.timeout_ok) + self.timeout_eps
            lock.extend(ttl, replace_ttl=True)
            query_res = worker.query(route_name, updated_params, timeout=timeout)
            gen = self.get_generation(session.pet_idx)
            res_update = self._after_pet_call(
                params,
                updated_params,
                session=session,
                route_name=route_name,
                res=query_res,
                gen=gen,
                timeout=timeout,
            )

            return Response(request_id, res_update.to_json())