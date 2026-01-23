from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from typing import List, Union, Any, Dict, Optional

from pytanque.routes import RouteName
from pytanque.client import Params, State
from redis import Redis

@dataclass
class StateExtended(State):
    generation: int

    @property
    def key(self) -> str:
        return f"{self.generation}:{self.st}"
    
    @classmethod
    def from_state(cls, state:State, generation:int):
        return cls(
            **asdict(state),
            generation=generation
        )

    def to_state(self) -> State:
        State.from_json(self.to_json())

    @classmethod
    def from_json(
        cls,
        x: Any
    ):
        state = State.from_json(x)
        return StateExtended.from_state(state, x['generation'])

    def to_json(self):
        state_raw = super().to_json()
        state_raw['generation'] = self.generation
        return state_raw
    
class RedisSerializable:
    redis_key: str

    def to_redis(self, session: Session, redis: Redis) -> None:
        key = f"{self.redis_key}:{session.id}"
        redis.set(key, json.dumps(self.to_json()))

    @classmethod
    def from_redis(
        cls,
        session: Session,
        redis: Redis
    ):
        key = f"{cls.redis_key}:{session.id}"
        raw = redis.get(key)
        if raw is None:
            return None
        return cls.from_json(json.loads(raw))

@dataclass
class QueryKwargs:
    route_name: RouteName
    params: Params
    timeout: float

    @classmethod
    def from_json(cls, data: dict) -> "QueryKwargs":
        return cls(
            route_name=RouteName(data["route_name"]),
            params=Params.from_json(data["params"]),
            timeout=float(data["timeout"]),
        )

    def to_json(self) -> dict:
        return {
            "route_name": self.route_name.value,
            "params": self.params.to_json(),
            "timeout": self.timeout,
        }

@dataclass
class TacticsParent(RedisSerializable):
    """
    Parent node, associated to a set of params to generate it.
    """
    state_key: str
    query_kwargs: QueryKwargs
    children: List[TacticsTree]=field(default_factory=list)
    redis_key = "tactics_tree"

    def find_node(self, state_key: str) -> TacticsParent | TacticsTree:
        if self.state_key == state_key:
            return self
        stack = list(self.children)
        while stack:
            node = stack.pop()
            if node.state_key == state_key:
                return node
            stack.extend(node.children)
        raise Exception("State not found")

    def find_path(self, state: StateExtended) -> list[TacticsTree]:
        state_key = state.key
        node = self.find_node(state_key)
        return node.trace_ancestors(node)

    def to_json(self) -> Any:
        return {
            "state_key": self.state_key,
            "query_kwargs": self.query_kwargs.to_json(),
            "children": [child.to_json() for child in self.children],
        }
    
    @classmethod
    def from_json(cls, data: dict) -> TacticsParent:
        parent = cls(
            state_key=data.get("state_key"),
            query_kwargs=QueryKwargs.from_json(data["query_kwargs"]),
            children=[]
        )

        for child_data in data.get("children", []):
            child = TacticsTree.from_json(child_data, parent=parent)
            parent.children.append(child)

        return parent

@dataclass
class TacticsTree:
    """
    Basic Tree structure to keep tracks of tactics.
    Beware of infinite recursion.
    """
    state_key: str
    tactic: str
    parent: Union[TacticsParent, TacticsTree]
    children: List[TacticsTree]=field(default_factory=list)

    def add_child(self, child: TacticsTree) -> None:
        child.parent = self
        self.children.append(child)

    def create_child(self, state_key: str, tactic: str) -> TacticsTree:
        child = TacticsTree(
            state_key=state_key,
            tactic=tactic,
            parent=self
        )
        self.children.append(child)
        return child
    
    def trace_ancestors(self) -> list[TacticsTree]:
        path = []
        node = self
        while isinstance(node, TacticsTree):
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def to_json(self):
        return {
            "state_key": self.state_key,
            "tactic": self.tactic,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_json(
        cls,
        data: dict,
        parent: TacticsParent | TacticsTree | None = None
    ) -> TacticsTree:
        node = cls(
            state_key=data["state_key"],
            tactic=data["tactic"],
            parent=parent,
            children=[]
        )

        for child_data in data.get("children", []):
            child = cls.from_json(child_data, parent=node)
            node.children.append(child)

        return node

@dataclass
class MappingState(RedisSerializable):
    mapping: Dict[str, StateExtended]
    redis_key = "mapping_state"

    @classmethod
    def from_json(cls, x:dict) -> MappingState:
        return cls({
            k:StateExtended.from_json(v) for k,v in x.items()
        })
    
    def to_json(self) -> Any:
        return {
            "mapping": {k: v.to_json() for k,v in self.mapping.items()}
        }
    
    def _key(self, state_or_key: Union[StateExtended, str]) -> str:
        return state_or_key if isinstance(state_or_key, str) else state_or_key.key

    def __getitem__(self, state_or_key: Union[StateExtended, str]) -> StateExtended:
        state_key = self._key(state_or_key)
        return self.mapping[state_key]
    
    def __contains__(self, state_or_key: Union[StateExtended, str]):
        state_key = self._key(state_or_key)
        return state_key in self.mapping

    def get(self, state_or_key: Union[StateExtended, str], default: Optional[StateExtended] = None):
        state_key = self._key(state_or_key)
        return self.mapping.get(state_key, default)

    def add(self, state: StateExtended):
        state_key = state.key
        self.mapping[state_key] = state

@dataclass
class Session:
    id: str
    pet_idx: int                       # which pet-server index (0..num_pet_server-1)
    redis_key = "session"

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> Session:
        return cls(
            id=raw["id"],
            pet_idx=raw["pet_idx"],
        )

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "pet_idx": self.pet_idx,
        }
    
    def to_redis(self, redis: Redis) -> None:
        key = f"{self.redis_key}:{self.id}"
        redis.set(key, json.dumps(self.to_json()))

    @classmethod
    def from_redis(
        self,
        session_id: int,
        redis: Redis
    ) -> Session:
        key = f"{self.redis_key}:{session_id}"
        raw = redis.get(key)
        if raw is None:
            return None
        return self.from_json(json.loads(raw))