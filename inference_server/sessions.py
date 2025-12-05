from dataclasses import dataclass, asdict
import time
from typing import List, Optional, Dict, Tuple
import os
import json
import uuid
import signal

from pytanque import Pytanque, State, Goal
import redis
from redis.lock import Lock

@dataclass
class Session:
    session_id: str
    pet_idx: int               # which pet-server index (0..num_pet_server-1)
    filepath: Optional[str]
    line: Optional[int]
    character: Optional[int]
    tactics: List[str]
    generation: Optional[int]        # pet-server generation where cached state is valid

class SessionManager:

    def __init__(self, pet_server_start_port: int=8765, num_pet_server: int=8, timeout_start_thm: int=30, timeout_run: int=10, timeout_kill_pet: int=5):
        self.redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        ports = [pet_server_start_port + k for k in range(num_pet_server)]
        self.pytanques = [Pytanque("127.0.0.1", port) for port in ports]
        for pet in self.pytanques:
            pet.connect()
        self.sessions: Dict[str, Session] = {} # session_id -> Session
        self.assigned_idx = 0 # for load balancing
        self.num_pet_server = num_pet_server
        self.timeout_start_thm = timeout_start_thm
        self.timeout_run = timeout_run
        self.timeout_kill_pet = timeout_kill_pet
        self.max_timeout = max(timeout_start_thm, timeout_run, timeout_kill_pet)

    @staticmethod
    def session_key(session_id: str) -> str:
        """Generate Redis key for a given session ID."""
        return f"session:{session_id}"

    @staticmethod
    def current_generation(pet_idx: str) -> str:
        """Generate Redis key for a given session ID."""
        return f"generation:{pet_idx}"

    @staticmethod
    def handler(signum, frame):
        """Signal handler for timeouts."""
        raise TimeoutError("Operation timed out")
    
    @staticmethod
    def pet_lock_key(pet_idx: int) -> str:
        return f"pet_lock:{pet_idx}"

    @staticmethod
    def pet_status_key(pet_idx: int) -> str:
        return f"pet_status:{pet_idx}"

    def archive_session(self, sess: Session):
        """Store session data in Redis for archival purposes."""
        self.redis_client.rpush("archived_sessions", json.dumps(asdict(sess)))

    def pet_status(self) -> bool:
        """Check if all pet-servers are in OK state."""
        for pet_idx in range(self.num_pet_server):
            state = self.redis_client.get(self.pet_status_key(pet_idx))
            if state is None or state.decode() != "OK":
                return False
        return True

    def ensure_pet_ok(self, pet_idx: int, timeout=5, poll_interval=0.1):
        """Ensure that the pet-server at pet_idx is in OK state."""
        state = self.redis_client.get(self.pet_status_key(pet_idx))
        if state is None:
            raise KeyError(f"pet_idx {pet_idx} has no status")
        
        t = time.time()
        while time.time() - t < timeout:
            state = self.redis_client.get(self.pet_status_key(pet_idx))
            if state is not None and state.decode() == "OK":
                return
            time.sleep(poll_interval)
        raise RuntimeError(f"pet_idx {pet_idx} not available (state={state})")

    def acquire_pet_lock(self, pet_idx: int, timeout: int=10) -> Lock:
        """
        Acquire a Redis lock for a given pet_idx.
        Returns the lock object (already acquired) or raises on failure.
        """
        lock = self.redis_client.lock(
            self.pet_lock_key(pet_idx),
            timeout=timeout,
            blocking=True
        )
        acquired = lock.acquire()
        if not acquired:
            raise TimeoutError(f"pet_idx {pet_idx} is busy")
        return lock

    def create_session(self) -> Session:
        """Create a new session with load-balanced pet-server assignment."""
        assigned_idx = self.assigned_idx
        self.assigned_idx = (self.assigned_idx + 1) % self.num_pet_server
        uid = str(uuid.uuid4())
        sess = Session(uid, assigned_idx, None, None, None, [], None)
        self.save_session(sess)
        self.sessions[uid] = sess
        return sess
    
    def get_session(self, session_id: str) -> Session:
        """Load session from Redis by session ID."""
        data = self.redis_client.get(self.session_key(session_id))
        if not data:
            raise KeyError("Unknown session_id")
        raw = json.loads(data)
        return Session(**raw)
    
    def save_session(self, sess: Session):
        """Save session to Redis."""
        self.redis_client.set(self.session_key(sess.session_id), json.dumps(asdict(sess)))

    def get_generation(self, pet_idx: str) -> int:
        """Load session from Redis by session ID."""
        data = self.redis_client.get(self.current_generation(pet_idx))
        if not data:
            raise KeyError("Unknown session_id")
        return int(data)

    def save_generation(self, session_id: str, generation: int):
        """Save session to Redis."""
        self.redis_client.set(self.current_generation(session_id), str(generation))

    def check_session(self, sess: Session):
        """If sess.generation is outdated, replay the tactics to recreate cache states on current pet-server generation."""
        current_generation = self.get_generation(sess.pet_idx)
        if sess.generation == current_generation:
            return  # No need to replay

        worker = self.pytanques[sess.pet_idx]

        state = None
        if sess.filepath:
            state = worker.get_state_at_pos(sess.filepath, sess.line, sess.character)

        for tactic in sess.tactics:
            state = worker.run(state, tactic, verbose=False, timeout=self.timeout_run)

        sess.generation = current_generation
        self.save_session(sess)  # Update session with new generation
    
    def send_kill_signal(self, pet_idx: int):
        """Send a kill signal to the pet-server at pet_idx."""
        self.redis_client.set(self.pet_status_key(pet_idx), "RESTART_NEEDED")
        
    def start_thm(self, session_id: str, filepath: str, line: int, character: int) -> Tuple[State, List[Goal]]:
        """Start a theorem proving session for the theorem at the given position."""
        sess = self.get_session(session_id)
        pet_idx = sess.pet_idx

        self.ensure_pet_ok(pet_idx, timeout=self.timeout_kill_pet)
        lock = self.acquire_pet_lock(pet_idx, timeout=self.max_timeout + 5)
        try:
            sess.generation = self.get_generation(sess.pet_idx)
            self.check_session(sess)
            worker = self.pytanques[pet_idx]

            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout_start_thm)

            state = worker.get_state_at_pos(filepath, line, character)
            goals = worker.goals(state)
            signal.alarm(0)  # Disable alarm

            if sess.filepath and sess.tactics:
                # archive session if non trivial previous session
                self.archive_session(sess)
            sess.filepath = filepath
            sess.line = line
            sess.character = character
            sess.tactics = []

            self.save_session(sess)
            return state, goals
        except TimeoutError:
            self.send_kill_signal(pet_idx)
            raise TimeoutError("Start theorem timed out; kill signal sent to pet-server.")
        finally:
            try:
                lock.release()
            except redis.exceptions.LockError:
                # Lock might have expired; safe to ignore
                pass

    def run(self, session_id: str, state: State, tactic: str) -> Tuple[State, List[Goal]]:
        """Execute a given tactic on the current proof state."""
        sess = self.get_session(session_id)
        pet_idx = sess.pet_idx

        self.ensure_pet_ok(pet_idx, timeout=self.timeout_kill_pet)
        lock = self.acquire_pet_lock(pet_idx, timeout=self.max_timeout + 5)
        try:
            self.check_session(sess)
            
            worker = self.pytanques[pet_idx]
            
            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(self.timeout_run)
            state = worker.run(state, tactic, verbose=False, timeout=self.timeout_run)
            goals = worker.goals(state)
            signal.alarm(0)

            sess.tactics.append(tactic)
            self.save_session(sess)

            return state, goals
        except TimeoutError:
            self.send_kill_signal(pet_idx)
            raise TimeoutError("Tactic execution timed out; kill signal sent to pet-server.")
        finally:
            try:
                lock.release()
            except redis.exceptions.LockError:
                # Lock might have expired; safe to ignore
                pass