from enum import StrEnum
from typing import List

class PetStatus(StrEnum):
    OK = "OK"
    RESTART_NEEDED = "RESTART_NEEDED"
    RESTARTING = "RESTARTING"

def session_key(session_id: str) -> str:
    return f"session:{session_id}"

def mapping_state_key(session_id: str) -> str:
    return f"mapping_state:{session_id}"

def tactics_tree_key(session_id: str) -> str:
    return f"tactics_tree_key:{session_id}"

def pet_status_key(pet_idx: int) -> str:
    return f"pet_status:{pet_idx}"

def generation_key(pet_idx: int) -> str:
    return f"generation:{pet_idx}"

def pet_lock_key(pet_idx: int) -> str:
    return f"pet_lock:{pet_idx}"

def monitor_epoch_key(pet_idx: int) -> str:
    return f"pet_monitor_epoch:{pet_idx}"

def session_assigned_idx_key() -> str:
    return "session_assigned_idx_key"

def archived_sessions_key() -> str:
    return "archived_sessions"

ALL_KEYS_STAR = [
    session_key('*'),
    mapping_state_key('*'),
    tactics_tree_key('*'),
    pet_status_key('*'),
    generation_key('*'),
    pet_lock_key('*'),
    monitor_epoch_key('*'),
    session_assigned_idx_key(),
    archived_sessions_key()
]
    
