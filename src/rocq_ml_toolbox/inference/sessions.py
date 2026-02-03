from __future__ import annotations
import time
from typing import List, Optional, Dict, Tuple, Any, Iterator, cast
from contextlib import contextmanager
from functools import singledispatchmethod, wraps
from dataclasses import fields
import logging
import json
import uuid


from pytanque import Pytanque, PetanqueError, PytanqueMode, Response
from pytanque.routes import Params, PETANQUE_ROUTES, UniversalRoute, BaseRoute, SessionRoute, InitialSessionRoute, Routes, Responses, RouteName

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
from .session_model import ParamsTree, MappingState, MappingTree, Session, State, QueryKwargs

logger = logging.getLogger("session")
profiling_logger = logging.getLogger("profiling")

def require_session_route(route: Routes, **kwargs) -> SessionRoute:
    if not isinstance(route, SessionRoute):
        raise SessionManagerError(
            f"expects SessionRoute, got {type(route).__name__}"
        )
    return cast(SessionRoute, route)

def normalize_payload(obj):
    """
    A bit hacky, at some point every response will be a full dataclass.
    """
    if hasattr(obj, "to_json") and callable(obj.to_json):
        return obj.to_json()
    return obj

def log_timing(name: str | None = None):
    def decorator(fn):
        fn_name = name or fn.__qualname__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                profiling_logger.info("%s took %.3f ms", fn_name, elapsed * 1000)
        return wrapper
    return decorator

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
        self.mappings_tree_cache: Dict[str, MappingTree] = {}
        self.params_trees_cache: Dict[str, Dict[str, ParamsTree]] = {}
        self.num_pet_server = num_pet_server
        self.timeout_ok = timeout_ok
        self.timeout_eps = timeout_eps

    def get_generation(self, pet_idx: int) -> int:
        data = self.redis_client.get(generation_key(pet_idx))
        if not data:
            raise SessionManagerError("Unknown session_id")
        return int(data)

    @log_timing()
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

        worker = Pytanque("127.0.0.1", self.ports[pet_idx], mode=PytanqueMode.SOCKET)
        worker.connect()
        self.pytanques[pet_idx] = worker
        self.worker_generations[pet_idx] = current_gen
        return worker

    def _restart_worker(self, pet_idx: int) -> Pytanque:
        """Remove worker at pet_idx"""
        if self.pytanques[pet_idx]:
            self.pytanques[pet_idx].close()
            self.pytanques[pet_idx] = None

    def archive_session(self, session: Session, params_tree: ParamsTree):
        """Store session data in Redis for archival purposes."""
        entry = {
            "session": session.to_json(),
            "params_tree": params_tree.to_json()
        }
        self.redis_client.rpush(archived_sessions_key(), json.dumps(entry))

    def pet_status(self) -> bool:
        """Check if all pet-servers are in OK state."""
        for pet_idx in range(self.num_pet_server):
            state = self.redis_client.get(pet_status_key(pet_idx))
            if state is None or state.decode() != PetStatus.OK:
                return False
        return True

    @log_timing()
    def ensure_pet_ok(self, pet_idx: int, timeout=15):
        req_id = str(uuid.uuid4())
        reply_channel = f"arbiter:reply:{pet_idx}:{req_id}"

        ps = self.redis_client.pubsub(ignore_subscribe_messages=True)
        ps.subscribe(reply_channel)

        try:
            req = {"id": req_id, "reply_to": reply_channel}
            self.redis_client.publish(f"arbiter:req:{pet_idx}", json.dumps(req))

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                msg = ps.get_message(timeout=1.0)
                if not msg:
                    continue
                if msg["type"] != "message":
                    continue

                resp = json.loads(msg["data"])
                if resp.get("id") == req_id:
                    # optionally validate resp["state"] etc.
                    return

            # timed out waiting for arbiter reply; decide based on state key
            state = self.redis_client.get(pet_status_key(pet_idx))
            raise SessionManagerError(
                f"pet_idx {pet_idx} not available (no arbiter reply, state={state})",
                require_restart=True,
            )
        finally:
            try:
                ps.unsubscribe(reply_channel)
            finally:
                ps.close()


    @log_timing()
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
        session = Session(pet_idx=assigned_idx)
        mapping_state = MappingState()
        mapping_tree = MappingTree()

        self.sessions_cache[session.id] = session
        self.mappings_state_cache[session.id] = mapping_state
        self.mappings_tree_cache[session.id] = mapping_tree
        self.params_trees_cache[session.id] = {}

        session.to_redis(self.redis_client)
        mapping_state.to_redis(session, self.redis_client)
        mapping_tree.to_redis(session, self.redis_client)
        return session.id
    
    @log_timing()
    def mapping_state_cache_update(self, state: State, session: Session) -> MappingState:
        """
        Update mapping_state_cache if the state is both outdated and not in it.
        """
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return mapping_state
        
        # check if session is in mappings_state_cache
        if session.id not in self.mappings_state_cache:
            mapping_state = MappingState.from_redis(session, self.redis_client)
            self.mappings_state_cache[session.id] = mapping_state
        elif state not in self.mappings_state_cache[session.id]:
            mapping_state = MappingState.from_redis(session, self.redis_client)
            self.mappings_state_cache[session.id] = mapping_state

        mapping_state = self.mappings_state_cache[session.id]
        return mapping_state
    
    @log_timing()
    def mapping_tree_cache_update(self, state: State, session: Session) -> MappingTree:
        """
        Update mappings_tree_cache if the state is not in it.
        """

        if session.id not in self.mappings_tree_cache:
            mapping_tree_cache = MappingTree.from_redis(session, self.redis_client)
            self.mappings_tree_cache[session.id] = mapping_tree_cache
        elif state not in self.mappings_tree_cache[session.id]:
            mapping_tree_cache = MappingTree.from_redis(session, self.redis_client)
            self.mappings_tree_cache[session.id] = mapping_tree_cache

        mapping_tree_cache = self.mappings_tree_cache[session.id]
        if state not in mapping_tree_cache:
            raise PetanqueError(-1, "State not found in updated mapping_tree_cache")
        return mapping_tree_cache

    @log_timing()
    def params_tree_cache_update(self, state: State, session: Session) -> ParamsTree:
        """
        Update params_trees_cache if the state is not in it.
        """
        mapping_tree = self.mapping_tree_cache_update(state, session)
        tree_id = mapping_tree[state]

        if session.id not in self.params_trees_cache:
            params_tree_cache = ParamsTree.from_redis(session, tree_id, self.redis_client)
            self.params_trees_cache[session.id] = {tree_id: params_tree_cache}
        elif tree_id not in self.params_trees_cache[session.id]:
            params_tree_cache = ParamsTree.from_redis(session, tree_id, self.redis_client)
            self.params_trees_cache[session.id][tree_id] = params_tree_cache
        elif state not in self.params_trees_cache[session.id][tree_id]:
            params_tree_cache = ParamsTree.from_redis(session, tree_id, self.redis_client)
            self.params_trees_cache[session.id][tree_id] = params_tree_cache
        
        params_tree_cache = self.params_trees_cache[session.id][tree_id]
        if state not in params_tree_cache:
            raise PetanqueError(-1, "State not found in updated params_tree_cache")
        return params_tree_cache

    @log_timing()
    def update_state(self, state: State, route: Routes, session: Session, lock:Lock, timeout_run=60, timeout_get_state=120) -> State:
        """If state.generation is outdated, replay the tactics to recreate cache states on current pet-server."""
        current_generation = self.get_generation(session.pet_idx)
        if state.generation == current_generation:
            return state # No need to replay
        worker = self._get_worker(session.pet_idx)

        mapping_state = self.mapping_state_cache_update(state, session)
        if state in mapping_state:
            state_map = mapping_state[state]
            if state_map.generation == current_generation:
                return state_map
        
        params_tree = self.params_tree_cache_update(state, session)
        lock.extend(timeout_get_state+self.timeout_eps, replace_ttl=True)
        replay_session = params_tree.find_path(state)
        logging.info(f"[{session.id}] State inconsistency, replay mechanism ON.")
        for node in replay_session:
            logging.info(f"[{session.id}] REPLAY: {node.query_kwargs.params}")
            state = mapping_state.get(node.state_key, None)
                
            # if state is outdated or None then regenerate it
            if not state or state.generation < current_generation:
                query_kwargs = node.query_kwargs.from_json(node.query_kwargs.to_json())
                query_kwargs.params = self._update_params(query_kwargs.params, route, session, lock)

                query_res = worker.query(**vars(query_kwargs))
                route = require_session_route(route)

                state = route.extract_response(query_res)
                state.generation = current_generation
                mapping_state.add(node.state_key, state)
        mapping_state.to_redis(session,self.redis_client)
        return state
    
    def send_kill_signal(self, pet_idx: int):
        """Send a kill signal to the pet-server at pet_idx."""
        self.redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
        self._restart_worker(pet_idx)

    @log_timing()
    def _update_params(
            self,
            params: Params,
            route: Routes,
            session: Session,
            lock: Lock
    ):
        new_params = params.from_json(params.to_json())
        for field in fields(new_params):
            state = getattr(new_params, field.name)
            if isinstance(state, State):
                new_state = self.update_state(state, route, session, lock)
                setattr(new_params, field.name, new_state)
        return new_params

    @contextmanager
    def _pet_ctx(
        self,
        session_id: str,
        route: Routes,
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
            updated_params = self._update_params(params, route, session, lock)
            yield session, worker, lock, updated_params

        except PetanqueError as e:
            # if petanque error is related to a timeout, then send kill signal to the underlying pet server.
            if e.code == -33_000:
                self.send_kill_signal(pet_idx)
                logging.warning(f"[{session.id}] Kill signal send to {pet_idx} after {params}, cause: {e}")
            raise e
        except SessionManagerError as e:
            if e.require_restart:
                self.send_kill_signal(pet_idx)
                logging.warning(f"[{session.id}] Kill signal send to {pet_idx} after {params}, cause: {e}")
            raise e
        except Exception as e:
            # if unknown issue then send kill signal to the underlying pet server.
            self.send_kill_signal(pet_idx)
            logging.warning(f"[{session.id}] Kill signal send to {pet_idx} after {params}, cause: {e}")
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
        route: Routes,
        params: Params,
        updated_params: Params,
        *,
        session: Session,
        route_name: RouteName,
        res: Responses,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        return res

    @_after_pet_call.register
    @log_timing('after_pet_call session case')
    def _(
        self,
        route: SessionRoute,
        params: Params,
        updated_params: Params,
        *,
        session: Session,
        route_name: RouteName,
        res: Responses,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        state = route.extract_response(res)
        state.generation = gen

        parent_state = route.extract_parent(params)
        mapping_tree = self.mapping_tree_cache_update(parent_state, session)
        tree_id = mapping_tree[parent_state]

        params_tree = ParamsTree.from_redis(session, tree_id, self.redis_client)
        parent_node = params_tree.find_node(parent_state)

        # we keep track of "old" client side params/state
        query_args = QueryKwargs(route_name, params, timeout=timeout)
        child = ParamsTree.from_state(state, query_args)
        parent_node.add_child(child)
        params_tree.to_redis(session, self.redis_client)

        self.mappings_tree_cache[session.id] = MappingTree.add_get_remote(state, params_tree, session, self.redis_client)
        return state

    @_after_pet_call.register
    @log_timing('after_pet_call primitive case')
    def _(
        self,
        route: InitialSessionRoute,
        params: Params,
        updated_params: Params,
        *,
        session: Session,
        route_name: RouteName,
        res: Responses,
        gen: int,
        timeout: Optional[float],
    ) -> None:
        state = route.extract_response(res)
        state.generation = gen
        
        query_args = QueryKwargs(route_name, params, timeout=timeout)
        params_tree = ParamsTree.from_state(state, query_args)
        params_tree.to_redis(session, self.redis_client)
        
        self.mappings_tree_cache[session.id] = MappingTree.add_get_remote(state, params_tree, session, self.redis_client)
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
        route = PETANQUE_ROUTES[route_name]
        with self._pet_ctx(session_id, route, params=params) as (session, worker, lock, updated_params):
            logging.info(f"[{session.id}] {route_name}: {params}")
            ttl = (timeout or self.timeout_ok) + self.timeout_eps
            lock.extend(ttl, replace_ttl=True)
            logging.info(f"[{session.id}] {updated_params}")
            query_res = worker.query(route_name, updated_params, timeout=timeout)
            logging.info(f"[{session.id}] {query_res}")
            if query_res is None:
                return Response(request_id, {})
            
            gen = self.get_generation(session.pet_idx)
            res_update = self._after_pet_call(
                route,
                params,
                updated_params,
                session=session,
                route_name=route_name,
                res=query_res,
                gen=gen,
                timeout=timeout,
            )

            return Response(request_id, normalize_payload(res_update))