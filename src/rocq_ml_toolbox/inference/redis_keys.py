from enum import StrEnum
from typing import List

class PetStatus(StrEnum):
    DOWN = "DOWN"
    STARTING = "STARTING"
    OK = "OK"
    RESTART_NEEDED = "RESTART_NEEDED"
    RESTARTING = "RESTARTING"

def session_key(session_id: str) -> str:
    return f"session:{session_id}"

def mapping_state_key(session_id: str) -> str:
    return f"mapping_state:{session_id}"

def mapping_tree_key(session_id: str) -> str:
    return f"mapping_tree:{session_id}"

def params_tree_key(session_id: str, tree_id: str) -> str:
    return f"params_tree:{session_id}:{tree_id}"

def tactics_tree_key(session_id: str) -> str:
    return f"tactics_tree_key:{session_id}"

def pet_status_key(pet_idx: int) -> str:
    return f"pet_status:{pet_idx}"

def generation_key(pet_idx: int) -> str:
    return f"generation:{pet_idx}"

def pet_lock_key(pet_idx: int) -> str:
    return f"pet_lock:{pet_idx}"

def pet_profile_key(pet_idx: int) -> str:
    return f"pet_profile:{pet_idx}"

def profile_bootstrap_key(profile: str) -> str:
    return f"profile_bootstrap:{profile}"

def monitor_epoch_key(pet_idx: int) -> str:
    return f"pet_monitor_epoch:{pet_idx}"

def session_assigned_idx_key() -> str:
    return "session_assigned_idx_key"

def archived_sessions_key() -> str:
    return "archived_sessions"

def arbiter_key() -> str:
    return f"arbiter"

def arbiter_heartbeat_key() -> str:
    return "arbiter:heartbeat"

ALL_KEYS_STAR = [
    session_key('*'),
    mapping_state_key('*'),
    mapping_tree_key('*'),
    params_tree_key('*', '*'),
    tactics_tree_key('*'),
    pet_status_key('*'),
    generation_key('*'),
    pet_lock_key('*'),
    pet_profile_key('*'),
    profile_bootstrap_key('*'),
    monitor_epoch_key('*'),
    session_assigned_idx_key(),
    archived_sessions_key(),
    arbiter_key(),
    arbiter_heartbeat_key(),
]
    
