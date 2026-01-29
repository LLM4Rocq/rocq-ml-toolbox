from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from typing import List, Union, Any, Dict, Optional, Self
import uuid
from abc import ABC, abstractmethod
from pytanque.routes import RouteName, PETANQUE_ROUTES
from pytanque.client import Params, State
from redis import Redis

def state_to_state_key(state: State) -> str:
    return f"{state.generation}:{state.st}"
    
class RedisSessionSerializable(ABC):
    redis_key: str
    def to_redis(self, session: Session, redis: Redis) -> None:
        key = f"{self.redis_key}:{session.id}"
        redis.set(key, json.dumps(self.to_json()))

    @classmethod
    def from_redis(
        cls,
        session: Session,
        redis: Redis
    ) -> Self:
        key = f"{cls.redis_key}:{session.id}"
        raw = redis.get(key)
        if raw is None:
            raise Exception(f'{cls.__name__} not found')
        return cls.from_json(json.loads(raw))

class RedisIDSerializable(ABC):
    redis_key: str
    id: str

    def to_redis(self, session: Session, redis: Redis) -> None:
        key = f"{self.redis_key}:{session.id}:{self.id}"
        redis.set(key, json.dumps(self.to_json()))

    @classmethod
    def from_redis(
        cls,
        session: Session,
        id: str,
        redis: Redis
    ) -> Self:
        key = f"{cls.redis_key}:{session.id}:{id}"
        raw = redis.get(key)
        if raw is None:
            raise Exception(f'{cls.__name__} not found')
        return cls.from_json(json.loads(raw))

@dataclass
class QueryKwargs:
    route_name: RouteName
    params: Params
    timeout: Optional[float]

    @classmethod
    def from_json(cls, data: dict) -> "QueryKwargs":
        route_name = RouteName(data["route_name"])
        params_cls = PETANQUE_ROUTES[route_name].params_cls
        return cls(
            route_name=route_name,
            params=params_cls.from_json(data["params"]),
            timeout=float(data["timeout"]) if data['timeout'] else None,
        )

    def to_json(self) -> dict:
        return {
            "route_name": self.route_name.value,
            "params": self.params.to_json(),
            "timeout": self.timeout,
        }

@dataclass
class ParamsTree(RedisIDSerializable):
    """
    Parent node, associated to a set of params to generate it.
    """
    state_key: str
    query_kwargs: QueryKwargs
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    redis_key: str = "params_tree"
    children: List[ParamsTree]=field(default_factory=list)
    parent: Optional[ParamsTree]=None

    @classmethod
    def from_state(
        cls,
        state,
        query_kwargs: QueryKwargs
    ) -> ParamsTree:
        return cls(state_key=state_to_state_key(state), query_kwargs=query_kwargs)

    def add_child(self, child: ParamsTree) -> None:
        child.parent = self
        self.children.append(child)

    def find_node(self, state: State) -> ParamsTree:
        state_key = state_to_state_key(state)
        if self.state_key == state_key:
            return self
        stack = list(self.children)
        while stack:
            node = stack.pop()
            if node.state_key == state_key:
                return node
            stack.extend(node.children)
        raise Exception("State not found")

    def __contains__(self, state: State) -> bool:
        try:
            self.find_node(state)
            return True
        except:
            return False
        
    def trace_ancestors(self) -> list[ParamsTree]:
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def find_path(self, state: State) -> list[ParamsTree]:
        node = self.find_node(state)
        return node.trace_ancestors()

    def to_json(self) -> Any:
        return {
            "state_key": self.state_key,
            "query_kwargs": self.query_kwargs.to_json(),
            "children": [child.to_json() for child in self.children],
            "id": self.id
        }
    
    @classmethod
    def from_json(cls, data: dict) -> ParamsTree:
        parent = cls(
            id=data['id'],
            state_key=data.get("state_key"),
            query_kwargs=QueryKwargs.from_json(data["query_kwargs"]),
            children=[]
        )

        for child_data in data.get("children", []):
            child = ParamsTree.from_json(child_data)
            child.parent = parent
            parent.children.append(child)

        return parent

@dataclass
class MappingState(RedisSessionSerializable):
    mapping: Dict[str, State]=field(default_factory=dict)
    redis_key: str ="mapping_state"

    @classmethod
    def from_json(cls, x:dict) -> MappingState:
        return cls({
            k:State.from_json(v) for k,v in x['mapping'].items()
        })
    
    def to_json(self) -> Any:
        return {
            "mapping": {k: v.to_json() for k,v in self.mapping.items()}
        }
    
    def _key(self, state_or_key: Union[State, str]) -> str:
        return state_or_key if isinstance(state_or_key, str) else state_to_state_key(state_or_key)

    def __getitem__(self, state_or_key: Union[State, str]) -> State:
        state_key = self._key(state_or_key)
        return self.mapping[state_key]
    
    def __contains__(self, state_or_key: Union[State, str]):
        state_key = self._key(state_or_key)
        return state_key in self.mapping

    def get(self, state_or_key: Union[State, str], default: Optional[State] = None):
        state_key = self._key(state_or_key)
        return self.mapping.get(state_key, default)

    def add(self, old_state_key: str, new_state: State):
        self.mapping[old_state_key] = new_state

@dataclass
class MappingTree(RedisSessionSerializable):
    mapping: Dict[str, str]=field(default_factory=dict)
    redis_key = "mapping_tree"
    
    @classmethod
    def from_json(cls, x:dict) -> MappingState:
        return cls(x['mapping'])
    
    def to_json(self) -> Any:
        return {
            "mapping": self.mapping
        }
    
    def _key(self, state_or_key: Union[State, str]) -> str:
        return state_or_key if isinstance(state_or_key, str) else state_to_state_key(state_or_key)

    def __getitem__(self, state_or_key: Union[State, str]) -> str:
        state_key = self._key(state_or_key)
        return self.mapping[state_key]
    
    def __contains__(self, state_or_key: Union[State, str]) -> bool:
        state_key = self._key(state_or_key)
        return state_key in self.mapping

    def add(self, state_or_key: Union[State, str], params_tree: ParamsTree):
        state_key = self._key(state_or_key)
        self.mapping[state_key] = params_tree.id

    @classmethod
    def add_get_remote(cls, state_or_key: Union[State, str], params_tree: ParamsTree, session: Session, redis: Redis) -> MappingTree:
        mapping_tree = MappingTree.from_redis(session, redis)
        mapping_tree.add(state_or_key, params_tree)
        mapping_tree.to_redis(session, redis)
        return mapping_tree

@dataclass
class Session(RedisSessionSerializable):
    pet_idx: int                       # which pet-server index (0..num_pet_server-1)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
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
        cls,
        session_id: int,
        redis: Redis
    ) -> Session:
        key = f"{cls.redis_key}:{session_id}"
        raw = redis.get(key)
        if raw is None:
            raise Exception(f'{cls.__name__} not found')
        return cls.from_json(json.loads(raw))