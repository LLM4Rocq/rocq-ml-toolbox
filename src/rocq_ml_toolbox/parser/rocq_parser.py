"""Tiny Rocq parser that replays proofs and records dependencies."""

from typing import List, Optional, Tuple, Generator, Dict, Union
import re

from pytanque.protocol import State, Opts

from .utils.ast import list_dependencies
from .utils.message import parse_about, parse_loadpath, solve_physical_path
from .utils.position import pos_to_offset, offset_to_pos

from .ast.model import VernacKind, VernacElement
from .parser import Range, Source, Dependency, Theorem, ParserError, Step
from ..inference.client import PetClient, ClientError



class RocqParser:
    """Interact with petanque to collect proof structure and metadata."""

    def __init__(self, client: PetClient):
        """Create a parser bound to a client."""
        super().__init__()
        self.client = client
        self.map_logical_physical = self._extract_loadpath()
        self.map_physical_logical = {v: k for k,v in self.map_logical_physical.items()}
    
    def add_logical_path(self, source: Source):
        source.logical_path = solve_physical_path(source.path, self.map_physical_logical)

    @staticmethod
    def _extract_blocks(content: str):
        return re.split(r'(?<=\.\s)', content)

    def extract_element(self, state: State, element: str, timeout: int=30, retry=1, is_notation=False) -> Tuple[Optional[VernacElement], bool]:
        try:
            if is_notation:
                substate = self.client.run(state, f'About "{element}".', timeout=timeout, retry=retry)
            else:
                substate = self.client.run(state, f'About {element}.', timeout=timeout, retry=retry)
            if substate.feedback:
                for feedback_tuple in substate.feedback:
                    feedback = feedback_tuple[1]
                    element, is_local = parse_about(feedback, self.map_logical_physical, self.map_physical_logical)
                    if element:
                        return element, is_local
                return None, False
        except ClientError:
            return None, False

    def extract_dependencies(self, state: State, tactic: str, timeout: int=30, retry=1) -> List[Dependency]:
        ast = self.client.ast(state, tactic)
        if ast:
            constants = list_dependencies(ast)
        else:
            constants = []
        dependencies = []
        for constant in constants:
            element, _ = self.extract_element(state, constant, timeout=timeout, retry=retry)
            if element:
                dependencies.append(element)
        return dependencies

    def _extract_loadpath(self, timeout: int=30, retry=1) -> Dict[str, str]:
        try:
            state = self.client.get_root_state('/tmp/init.v')
        except ClientError as e:
            raise ParserError('Missing /tmp/init.v file.')
        result = self.client.run(state, 'Print LoadPath.', timeout=timeout, retry=retry)
        return parse_loadpath(result.feedback[0][1])

    def extract_toc(self, source: Source) -> List[VernacElement]:
        toc = self.client.get_ast(source.path)
        content_utf_8 = source.content.encode("utf-8")
        for entry in toc:
            entry.data['content'] = content_utf_8[entry.span.bp:entry.span.ep].decode("utf-8")
        return toc

    def extract_proofs(self, source: Source, timeout=120, retry=1, solve_deps=False, verbose=False) -> Generator[Theorem, None, None]:
        ast = self.client.get_ast(source.path)
        proof_open = False
        initial_goals = None
        theorem_element = None
        steps = []
        namespaces_stack = []
        state = None
        for entry in ast:
            span = entry.span
            kind = entry.kind
            subcontent = source.content_utf8[span.bp:span.ep].decode("utf-8")
            match kind:
                case VernacKind.BEGIN_SECTION:
                    namespaces_stack.append(("SECTION", entry.name))
                case VernacKind.DECLARE_MODULE_TYPE| VernacKind.DEFINE_MODULE:
                    if not entry.data['is_alias']:
                        namespaces_stack.append(("MODULE", entry.name))
                case VernacKind.END_SEGMENT:
                    last_el = namespaces_stack.pop()
                    assert entry.name == last_el[1]
                case VernacKind.START_THEOREM_PROOF:
                    proof_open = True
                    theorem_element = entry
                    pos = offset_to_pos(source.content_utf8, entry.span.ep)
                    state = self.client.get_state_at_pos(source.path, pos.line, pos.character, timeout=timeout, retry=retry)
                    initial_goals = self.client.goals(state)
                case VernacKind.PROOF_STEP:
                    if proof_open:
                        deps = self.extract_dependencies(state, subcontent)
                        state = self.client.run(state, subcontent, timeout=timeout, retry=retry)
                        goals = self.client.goals(state)
                        step = Step(subcontent, goals, deps)
                        steps.append(step)
                        
                case VernacKind.END_PROOF:
                    if proof_open:
                        stack_modules = [el[1] for el in namespaces_stack if el[0] == 'MODULE']
                        stack_modules.append(theorem_element.name)
                        theorem_element.data['fqn'] = ".".join(stack_modules)
                        yield Theorem(steps, initial_goals, theorem_element)

            match kind:
                case VernacKind.START_THEOREM_PROOF | VernacKind.PROOF | VernacKind.PROOF_STEP:
                    pass
                case _:
                    proof_open = False
                    steps = []
                    initial_goals = None
                    theorem_element = None