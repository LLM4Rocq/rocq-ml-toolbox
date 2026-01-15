from __future__ import annotations
from dataclasses import dataclass, asdict
import time
from typing import List, Optional, Dict, Tuple, Any, Callable, Iterator, TypeVar
from contextlib import contextmanager

T = TypeVar("T")

import json
import uuid
import signal

from pytanque import Pytanque, PetanqueError
from pytanque.protocol import (
    Opts,
    State,
    Goal,
    Inspect
)

import redis
from redis.lock import Lock

from .client import StateExtended
from .redis_keys import (
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
class Session:
    session_id: str
    pet_idx: int                       # which pet-server index (0..num_pet_server-1)
    filepath: Optional[str]
    line: Optional[int]
    character: Optional[int]
    tactics: List[Tuple[StateExtended, str]]
    generation: Optional[int]          # pet-server generation where cached state is valid
    mapping_state: Dict[str, StateExtended]  # to map old states with new states in case of replay

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> Session:
        tactics = [(StateExtended.from_json(st_json), tac) for st_json, tac in raw.get("tactics", [])]
        mapping_state_raw = raw.get("mapping_state", {})
        mapping_state = {id_str: StateExtended.from_json(v_json) for id_str, v_json in mapping_state_raw.items()}

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

    def to_json(self) -> dict:
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

    def __init__(self, redis_url: str, pet_server_start_port: int=8765, num_pet_server: int=4, timeout_ok: int=15, timeout_eps: int=10):
        self.redis_client = redis.Redis.from_url(redis_url)
        self.ports = [pet_server_start_port + k for k in range(num_pet_server)]
        self.pytanques: List[Optional[Pytanque]] = [None] * num_pet_server
        self.worker_generations: List[Optional[int]] = [None] * num_pet_server
        self.sessions: Dict[str, Session] = {} # session_id -> Session
        self.num_pet_server = num_pet_server
        self.timeout_ok = timeout_ok
        self.timeout_eps = timeout_eps

    @staticmethod
    def handler(signum, frame):
        """Signal handler for timeouts."""
        raise UnresponsiveError("Operation timed out")

    def save_session(self, sess: Session):
        """Save session to Redis."""
        self.redis_client.set(
            session_key(sess.session_id),
            json.dumps(sess.to_json())
        )

    def get_session(self, session_id: str) -> Session:
        """Load session from Redis by session ID."""
        data = self.redis_client.get(session_key(session_id))
        if not data:
            raise KeyError("Unknown session_id")
        raw = json.loads(data)
        return Session.from_json(raw)

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

    def _get_state_at_pos(self, pet_idx: int, filepath: str, line: int, character: int, opts: Optional[Opts]=None) -> StateExtended:
        """get_state_at_pos wrapper to cache state."""
        worker = self._get_worker(pet_idx)
        generation = self.get_generation(pet_idx)
        id_str = str({
            "filepath": filepath,
            "line": line,
            "character": character,
            "generation": generation,
            "pet_idx": pet_idx
        })
        raw = self.redis_client.get(cache_state_key(id_str))
    
        state_ext: Optional[StateExtended] = None
        if not raw:
            state = worker.get_state_at_pos(filepath, line, character, opts)
            state_ext = StateExtended.from_state(state, generation)
        else:
            data = json.loads(raw)
            cache_state_ext = StateExtended.from_json(data)
            if cache_state_ext.generation != generation:
                state = worker.get_state_at_pos(filepath, line, character, opts)
                state_ext = StateExtended.from_state(state, generation)
            else:
                state_ext = cache_state_ext

        self.redis_client.set(cache_state_key(id_str), json.dumps(state_ext.to_json()))
        return state_ext
        

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

    def create_session(self, timeout: int=10) -> str:
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
        return sess.session_id
    
    def check_session(self, sess: Session, lock:Lock, timeout_run=60, timeout_get_state=120) -> Session:
        """If sess.generation is outdated, replay the tactics to recreate cache states on current pet-server generation."""
        current_generation = self.get_generation(sess.pet_idx)
        if sess.generation == current_generation:
            return sess # No need to replay
        
        worker = self._get_worker(sess.pet_idx)
        state_ext = None
        if sess.filepath:
            try:
                signal.signal(signal.SIGALRM, SessionManager.handler)
                signal.alarm(timeout_get_state)
                lock.extend(timeout_get_state+self.timeout_eps, replace_ttl=True)
                state_ext = self._get_state_at_pos(sess.pet_idx, sess.filepath, sess.line, sess.character)
                for old_state_ext, tactic in sess.tactics:
                    signal.alarm(timeout_run)
                    lock.extend(timeout_run+self.timeout_eps, replace_ttl=True)
                    state = worker.run(state_ext, tactic, verbose=False, timeout=timeout_run)
                    state_ext = StateExtended.from_state(state, current_generation)

                    old_state_ext_str = old_state_ext.to_json_string()
                    sess.mapping_state[old_state_ext_str] = state_ext
            finally:
                signal.alarm(0)
        sess.generation = current_generation
        self.save_session(sess)  # Update session with new generation
        return sess
    
    def send_kill_signal(self, pet_idx: int):
        """Send a kill signal to the pet-server at pet_idx."""
        self.redis_client.set(pet_status_key(pet_idx), PetStatus.RESTART_NEEDED)
        self._restart_worker(pet_idx)

    def _failure_simulation(self):
        raise UnresponsiveError("Forced failure (Debugging purpose).")

    @contextmanager
    def _alarm(self, seconds: Optional[int]) -> Iterator[None]:
        if not seconds:
            yield
            return
        prev = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, SessionManager.handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)

    @contextmanager
    def _pet_ctx(
        self,
        session_id: str,
        failure: bool,
        error_prefix: str,
        state_ext: Optional[StateExtended]=None
    ) -> Iterator[Tuple[Session, Pytanque, Lock, Optional[StateExtended]]]:
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

            if failure:
                self._failure_simulation()

            yield sess, worker, lock, state_ext

        except KeyError:
            raise
        except PetanqueError:
            raise
        except ValueError:
            raise
        except Exception as e:
            self.send_kill_signal(pet_idx)
            raise UnresponsiveError(
                f"{error_prefix} ({e}); kill signal sent to pet-server."
            ) from e
        finally:
            signal.alarm(0)
            if lock is not None:
                try:
                    lock.release()
                except redis.exceptions.LockError:
                    pass

    def _pet_call(
        self,
        session_id: str,
        timeout: Optional[int],
        failure: bool,
        error_prefix: str,
        op: Callable[[Session, Pytanque, Lock, Optional[StateExtended]], T],
        state_ext: Optional[StateExtended] = None
    ) -> T:
        with self._pet_ctx(
            session_id,
            failure=failure,
            error_prefix=error_prefix,
            state_ext=state_ext
        ) as (sess, worker, lock, new_state_ext):
            with self._alarm(timeout):
                return op(sess, worker, lock, new_state_ext)

    def get_state_at_pos(self, session_id: str, filepath: str, line: int, character: int, opts: Optional[Opts]=None, failure: bool=False, timeout=120) -> StateExtended:
        """Start a theorem proving session for the theorem at the given position. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state: Optional[StateExtended]):
            sess.generation = self.get_generation(sess.pet_idx)

            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            state_ext = self._get_state_at_pos(sess.pet_idx, filepath, line, character, opts)
            if sess.tactics:
                self.archive_session(sess)
            
            sess.filepath = filepath
            sess.line = line
            sess.character = character
            sess.tactics = [(state_ext, "")]
            sess.mapping_state = {}
            self.save_session(sess)

            return state_ext
        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Start theorem failed",
            op=op,
        )

    def run(self, session_id: str, state_ext: StateExtended, tactic: str, opts: Optional[Opts]=None, failure: bool=False, timeout=60) -> StateExtended:
        """Execute a given tactic on the current proof state. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            new_state = worker.run(state_ext.to_state(), tactic, opts=opts, verbose=False, timeout=timeout)
            new_state_ext = StateExtended.from_state(new_state, state_ext.generation)

            sess.tactics.append((new_state_ext, tactic))
            self.save_session(sess)

            return new_state_ext

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Tactic execution failed",
            op=op,
            state_ext=state_ext
        )
    
    def goals(self, session_id: str, state_ext: StateExtended, pretty=True, failure=False, timeout=10) -> List[Goal]:
        """Gather goals associated to a state. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            goals = worker.goals(state_ext.to_state(), pretty=pretty)
            return goals

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Goals gathering failed",
            op=op,
            state_ext=state_ext
        )

    def complete_goals(self, session_id: str, state_ext: StateExtended, pretty=True, failure=False, timeout=10) -> List[Dict]:
        """Gather complete goals associated to a state. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            goals = worker.complete_goals(state_ext.to_state(), pretty=pretty)
            return goals

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Complete goals gathering failed",
            op=op,
            state_ext=state_ext
        )

    def premises(self, session_id: str, state_ext: StateExtended, failure=False, timeout=10) -> Any:
        """Gather accessible premises (lemmas, definitions) from a state. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            premises = worker.premises(state_ext.to_state())
            return premises

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Premises gathering failed",
            op=op,
            state_ext=state_ext
        )

    def state_equal(self, session_id: str, st1_ext: StateExtended, st2_ext: StateExtended, kind: Inspect, failure=False, timeout=10) -> bool:
        """Check whether st1 is equal to st2. Beware st1, and st2 are expected to belong to the same session. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            nonlocal st2_ext
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)

            st2_ext_str = st2_ext.to_json_string()
            if st2_ext_str in sess.mapping_state:
                st2_ext = sess.mapping_state[st2_ext_str]
            result = worker.state_equal(state_ext.to_state(), st2_ext.to_state(), kind)
            return result

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="State hash failed",
            op=op,
            state_ext=st1_ext
        )

    def state_hash(self, session_id: str, state_ext: StateExtended, failure=False, timeout=10) -> int:
        """Get a hash value for a proof state. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            hash = worker.state_hash(state_ext.to_state())
            return hash

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="State hash failed",
            op=op,
            state_ext=state_ext
        )

    def toc(self, session_id: str, file: str, failure=False, timeout=120) -> list[tuple[str, Any]]:
        """Get toc of a file. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            toc = worker.toc(file)
            return toc

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Toc failed",
            op=op
        )
    
    def ast(self, session_id: str, state_ext: StateExtended, text: str, failure=False, timeout=120) -> Dict:
        """Get ast of a command parsed at a state. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            ast = worker.ast(state_ext.to_state(), text)
            return ast

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="AST failed",
            op=op,
            state_ext=state_ext
        )

    def ast_at_pos(self, session_id: str, file: str, line: int, character: int, failure=False, timeout=120) -> Dict:
        """Get ast at a specified position in a file. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            ast = worker.ast_at_pos(file, line, character)
            return ast

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="AST at position failed",
            op=op
        )

    # def op(sess: Session, worker: Pytanque, lock:Lock, state: Optional[StateExtended]):
    #         sess.generation = self.get_generation(sess.pet_idx)

    #         lock.extend(timeout+self.timeout_eps, replace_ttl=True)
    #         state = self._get_state_at_pos(sess.pet_idx, filepath, line, character, opts)
    #         if sess.tactics:
    #             self.archive_session(sess)
            
    #         sess.filepath = filepath
    #         sess.line = line
    #         sess.character = character
    #         sess.tactics = [(state, "")]
    #         sess.mapping_state = {}
    #         self.save_session(sess)

    #         return state
    def get_root_state(self, session_id: str, file: str, opts: Optional[Opts]=None, failure=False, timeout=120) -> StateExtended:
        """Get root state of a document. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            sess.generation = self.get_generation(sess.pet_idx)
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            state = worker.get_root_state(file, opts=opts)
            state_ext = StateExtended.from_state(state, sess.generation)
            return state_ext

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Get root state failed",
            op=op
        )
    
    def list_notations_in_statement(self, session_id: str, state_ext: StateExtended, statement: str, failure=False, timeout=10) -> list[Dict]:
        """Get the list of notations appearing in a theorem/lemma statement. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            notations = worker.list_notations_in_statement(state_ext.to_state(), statement)
            return notations

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="List notations in statement failed",
            op=op,
            state_ext=state_ext
        )

    def start(self, session_id: str, file: str, thm: str, pre_commands: Optional[str]=None, opts: Optional[Opts]=None, failure=False, timeout=120) -> StateExtended:
        """Start a proof session for a specific theorem in a Coq/Rocq file. See pytanque documentation for more details."""
        def op(sess: Session, worker: Pytanque, lock:Lock, state_ext: StateExtended):
            gen = self.get_generation(sess.pet_idx)
            lock.extend(timeout+self.timeout_eps, replace_ttl=True)
            state = worker.start(file, thm, pre_commands=pre_commands, opts=opts)
            state_ext = StateExtended.from_state(state, gen)
            return state_ext

        return self._pet_call(
            session_id,
            timeout=timeout,
            failure=failure,
            error_prefix="Get root state failed",
            op=op
        )

    # def query(self, session_id: str, params: Params, size:int=4096, failure=False, timeout=120) -> Response:
    #     """Send a low-level JSON-RPC query to the server. See pytanque documentation for more details."""
    #     def op(sess: Session, worker: Pytanque, lock:Lock, state: State):
    #         lock.extend(timeout+self.timeout_eps, replace_ttl=True)
    #         resp = worker.query(params, size)
    #         return resp
        
    #     return self._pet_call(
    #         session_id,
    #         timeout=timeout,
    #         failure=failure,
    #         error_prefix="Get root state failed",
    #         op=op
    #     )