from enum import StrEnum

class PetStatus(StrEnum):
    OK = "OK"
    RESTART_NEEDED = "RESTART_NEEDED"
    RESTARTING = "RESTARTING"

def session_key(session_id: str) -> str:
    return f"session:{session_id}"

def cache_state_key(id: int) -> str:
    return f"cache_state:{id}"

def pet_status_key(pet_idx: int) -> str:
    return f"pet_status:{pet_idx}"

def generation_key(pet_idx: int) -> str:
    return f"generation:{pet_idx}"

def pet_lock_key(pet_idx: int) -> str:
    return f"pet_lock:{pet_idx}"

def monitor_epoch_key(pet_idx: int) -> str:
    return f"pet_monitor_epoch:{pet_idx}"

def session_lock_key() -> str:
    return "session_lock"

def session_assigned_idx_key() -> str:
    return "session_assigned_idx_key"

def archived_sessions_key() -> str:
    return "archived_sessions"
