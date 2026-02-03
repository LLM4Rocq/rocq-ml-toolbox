"""Tiny Rocq parser that replays proofs and records dependencies."""

from typing import List, Optional, Tuple, Generator, Dict, Union
import re
import tempfile

from pytanque.protocol import State, Opts

from .utils.ast import list_dependencies
from .utils.message import parse_about, parse_loadpath, solve_physical_path
from .utils.position import pos_to_offset, offset_to_pos

from .ast.model import VernacKind, VernacElement, Span
from .parser import Source, Theorem, ParserError, Step

from pytanque import PetanqueError
from ..inference.client import PytanqueExtended

class RocqParser:
    """Interact with petanque to collect proof structure and metadata."""

    def __init__(self, client: PytanqueExtended):
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

    def extract_element(self, state: State, element: str, timeout: int=30, is_notation=False) -> Tuple[Optional[VernacElement], bool]:
        try:
            if is_notation:
                substate = self.client.run(state, f'About "{element}".', timeout=timeout)
            else:
                substate = self.client.run(state, f'About {element}.', timeout=timeout)
            if substate.feedback:
                for feedback_tuple in substate.feedback:
                    feedback = feedback_tuple[1]
                    element, is_local = parse_about(feedback, self.map_logical_physical, self.map_physical_logical)
                    if element:
                        return element, is_local
                return None, False
        except PetanqueError:
            return None, False

    def extract_dependencies(self, state: State, tactic: str, timeout: int=30) -> List[VernacElement]:
        ast = self.client.ast(state, tactic)
        if ast:
            constants = list_dependencies(ast)
        else:
            constants = []
        dependencies = []
        for constant in constants:
            element, _ = self.extract_element(state, constant, timeout=timeout)
            if element:
                element.name = constant
                dependencies.append(element)
        return dependencies

    def _extract_loadpath(self, timeout: int=30) -> Dict[str, str]:
        path = self.client.empty_file()
        try:
            state = self.client.get_root_state(path)
        except PetanqueError as e:
            raise ParserError(f'Issue with temporary file: {path}')
        result = self.client.run(state, 'Print LoadPath.', timeout=timeout)
        return parse_loadpath(result.feedback[0][1])

    def extract_toc(self, source: Source) -> List[VernacElement]:
        toc = self.client.get_ast(source.path)
        content_utf_8 = source.content.encode("utf-8")
        for entry in toc:
            entry.data['content'] = content_utf_8[entry.span.bp:entry.span.ep].decode("utf-8")
        return toc
    
    def ast(self, source: Source) -> Tuple[List[VernacElement], List[VernacElement]]:
        ast = self.client.get_ast(source.path)
        target_elements = []
        proof_elements = []
        namespaces_stack = []

        targets_kind = [
            VernacKind.DEFINITION,
            VernacKind.SYNTACTIC_DEFINITION,
            VernacKind.START_THEOREM_PROOF,
            VernacKind.NOTATION,
            VernacKind.RESERVED_NOTATION,
            VernacKind.FIXPOINT,
            VernacKind.COFIXPOINT,
            VernacKind.COERCION,
            VernacKind.CANONICAL,
            VernacKind.INSTANCE,
            VernacKind.INDUCTIVE,
            VernacKind.COINDUCTIVE,
            VernacKind.RECORD,
            VernacKind.STRUCTURE,
            VernacKind.VARIANT,
            VernacKind.CLASS,
            VernacKind.LTAC,
            VernacKind.CONSTANT,
            VernacKind.FIELD,
            VernacKind.CONSTRUCTOR
        ]
        proofs_kind = [
            VernacKind.PROOF,
            VernacKind.PROOF_STEP,
            VernacKind.SUBPROOF,
            VernacKind.END_SUBPROOF,
            VernacKind.BULLET,
            VernacKind.PROOF,
            VernacKind.END_PROOF
        ]
        for entry in ast:
            kind = entry.kind
            match kind:
                case VernacKind.BEGIN_SECTION:
                    if entry.name:
                        namespaces_stack.append(("SECTION", entry.name))
                case VernacKind.DECLARE_MODULE_TYPE| VernacKind.DEFINE_MODULE:
                    if not entry.data['is_alias'] and entry.name:
                        namespaces_stack.append(("MODULE", entry.name))
                case VernacKind.END_SEGMENT:
                    last_el = namespaces_stack.pop()
                    assert entry.name == last_el[1]
                case _ if kind in targets_kind:
                    stack_modules = [el[1] for el in namespaces_stack if el[0] == 'MODULE']
                    history = [el[1] for el in namespaces_stack]
                    if not entry.name:
                        continue
                    stack_modules.append(entry.name)
                    full_name = ".".join(stack_modules)
                    entry.data['fqn'] = full_name
                    entry.data['history'] = ".".join(history)
                    target_elements.append(entry)
                case _ if kind in proofs_kind:
                    proof_elements.append(entry)
        return target_elements, proof_elements

    def extract_proofs_raw(self, source: Source) -> List[Tuple[VernacElement, List[str]]]:
        ast = self.client.get_ast(source.path)
        proof_open = False
        theorem_element = None
        steps = []
        result = []
        namespaces_stack = []
        prev_entry = None
        for entry in ast:
            span = entry.span
            kind = entry.kind
            subcontent = source.content_utf8[span.bp:span.ep].decode("utf-8")
            match kind:
                case VernacKind.BEGIN_SECTION:
                    namespaces_stack.append(("SECTION", entry.name))
                case VernacKind.DECLARE_MODULE_TYPE| VernacKind.DEFINE_MODULE:
                    if not entry.data['is_alias'] and entry.name:
                        namespaces_stack.append(("MODULE", entry.name))
                case VernacKind.END_SEGMENT:
                    last_el = namespaces_stack.pop()
                    assert entry.name == last_el[1]
                case VernacKind.START_THEOREM_PROOF:
                    proof_open = True
                    theorem_element = entry
                case VernacKind.PROOF:
                    if not prev_entry:
                        raise ParserError(f'No prev_entry at {entry.span}')
                    proof_open = True
                    theorem_element = prev_entry
                    steps.append(subcontent)
                case VernacKind.PROOF_STEP | VernacKind.SUBPROOF | VernacKind.END_SUBPROOF | VernacKind.BULLET:
                    if proof_open:
                        steps.append(subcontent)
                case VernacKind.END_PROOF:
                    if proof_open:
                        stack_modules = [el[1] for el in namespaces_stack if el[0] == 'MODULE']
                        if not theorem_element.name:
                            continue
                        stack_modules.append(theorem_element.name)
                        theorem_element.data['fqn'] = ".".join(stack_modules)
                        result.append((theorem_element, steps))
                    proof_open = False
                    steps = []
                    theorem_element = None
            match kind:
                case VernacKind.PROOF | VernacKind.PROOF_STEP | VernacKind.SUBPROOF | VernacKind.END_SUBPROOF | VernacKind.BULLET:
                    pass
                case VernacKind.START_THEOREM_PROOF:
                    steps = []
                case _:
                    proof_open = False
                    steps = []
                    theorem_element = None
            
            prev_entry = entry
        return result

    def extract_full_proof(self, source: Source, element: VernacElement, steps_raw: List[str], timeout=60):
        if not element.span:
            raise ParserError(f'{element} has no span in {source.path}')
        pos = offset_to_pos(source.content_utf8, element.span.ep)
        state = self.client.get_state_at_pos(source.path, pos.line, pos.character, timeout=timeout)
        initial_goals = self.client.goals(state)
        steps = []
        for step_str in steps_raw:
            deps = self.extract_dependencies(state, step_str)
            state = self.client.run(state, step_str, timeout=timeout)
            goals = self.client.goals(state)
            step = Step(step_str, goals, deps)
            steps.append(step)
        return Theorem(steps, initial_goals, element)