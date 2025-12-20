"""Tiny Rocq parser that replays proofs and records dependencies."""

from typing import List, Optional, Tuple, Generator, Dict, Union
import re
import json

from pytanque.protocol import State, TocElement

from .utils.ast import list_dependencies
from .utils.message import parse_about, parse_loadpath, solve_physical_path
from .utils.position import extract_subtext, move_position

from .parser import Step, Position, Range, Element, Source, Dependency, Theorem, ParserError, Notation
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

    def extract_element(self, state: State, element: str, timeout: int=30, retry=1, is_notation=False) -> Optional[Element]:
        if is_notation:
            substate = self.client.run(state, f'About "{element}".', timeout=timeout, retry=retry)
        else:
            substate = self.client.run(state, f'About {element}.', timeout=timeout, retry=retry)
        if substate.feedback:
            result = substate.feedback[0][1]
            return parse_about(result, self.map_logical_physical, self.map_physical_logical)
        return None

    def extract_dependencies(self, state: State, tactic: str, timeout: int=30, retry=1) -> List[Dependency]:
        ast = self.client.ast(state, tactic)
        if ast:
            constants = list_dependencies(ast)
        else:
            constants = []
        dependencies = []
        for constant in constants:
            element = self.extract_element(state, constant, timeout=timeout, retry=retry)
            if element:
                dependencies.append(element)
        return dependencies

    def _extract_loadpath(self, timeout: int=30, retry=1) -> Dict[str, str]:
        try:
            state = self.client.get_root_state('/tmp/init.v')
        except ClientError:
            raise ParserError('Missing /tmp/init.v file.')
        result = self.client.run(state, 'Print LoadPath.', timeout=timeout, retry=retry)
        return parse_loadpath(result.feedback[0][1])

    def extract_ast(self, source: Source, timeout: int=120, retry=1) -> Generator[Tuple[int, int, dict], None, None]:
        buffer = ""
        state = self.client.get_root_state(source.path, timeout=timeout, retry=retry)

        blocks = RocqParser._extract_blocks(source.content)
        curr_pos = Position(0, 0)
        for block in blocks:
            try:
                state = self.client.get_state_at_pos(source.path, curr_pos.line, curr_pos.character, timeout=timeout, retry=retry)
            except ClientError:
                pass
            except Exception as e:
                print(type(e))
                exit()
            buffer += block
            try:
                ast = self.client.ast(state, buffer, timeout=timeout, retry=retry)
                yield (curr_pos.line, curr_pos.character, ast)
                curr_pos = move_position(source.content, curr_pos, len(buffer))
                buffer = ""
            except ClientError:
                pass
    
    def extract_proofs(self, source: Source, timeout: int=120, retry=1) -> Generator[Tuple[Element, Range, List[str]], None, None]:
        ast = self.extract_ast(source, timeout=timeout, retry=retry)

        thm_name: Optional[str] = None
        start: Optional[Position] = None
        kind: Optional[str] = None
        range_statement: Optional[Range] = None
        for offset_line, offset_char, entry in ast:
            curr_pos = Position(offset_line, offset_char)
            try:
                node_start_char = entry['loc']['bp']
                node_end_char = entry['loc']['ep']
                content = entry['v']['expr'][1]

                node_start_pos = move_position(source.content, curr_pos, node_start_char)
                node_end_pos = move_position(source.content, curr_pos, node_end_char)
                # TODO: Extract Instance proof
                if content[0] == 'VernacStartTheoremProof':
                    infos = content[2][0][0][0]
                    name = infos['v'][1]
                    print(f"START PROOF {name}")
                    kind = content[1][0]
                    thm_name = name
                    start = node_end_pos
                    range_statement = Range(node_start_pos, node_end_pos)
                elif content[0] == 'VernacDefinition':
                    infos = content[2][0]
                    name = infos['v'][1][1]
                    print(f"START DEFINITION {name}")
                    kind = content[1][1][0]
                    thm_name = name
                    start = node_end_pos
                    range_statement = Range(node_start_pos, node_end_pos)
                elif content[0] == 'VernacInstance':
                    infos = content[1][0]
                    name = infos['v'][1][1]
                    print(f"START INSTANCE {name}")
                    kind = 'Instance'
                    thm_name = name
                    start = node_end_pos
                    range_statement = Range(node_start_pos, node_end_pos)
                elif content[0] == 'VernacEndProof':
                    end = node_end_pos
                    print(f"END PROOF {name}")
                    if not (start and end and thm_name):
                        print(f"Ignore {node_start_pos} in {source.path} (no VernacStartTheoremProof associated).")
                        continue
                    range_proof = Range(start, end)

                    proof = extract_subtext(source.content, range_proof)
                    statement = extract_subtext(source.content, range_statement)
                    proof_steps = RocqParser._extract_blocks(proof)

                    state = self.client.get_state_at_pos(source.path, end.line, end.character, timeout=timeout, retry=retry)
                    element = self.extract_element(state, thm_name, timeout=timeout, retry=retry)
                    assert element.path == source.path, f"Path mismatch between {element.path} and {source.path}"
                    element.content = statement
                    element.kind = kind
                    element.content_range = range_statement

                    thm_name: Optional[str] = None
                    start: Optional[Position] = None
                    kind: Optional[str] = None
                    range_statement: Optional[Range] = None
                    yield (element, range_proof, proof_steps)
            except (KeyError, TypeError) as e:
                pass

    def execute_proof(self, element: Element, proof_steps: List[str], timeout: int=120, retry=1) -> Theorem:
        filepath = element.path
        line = element.range.start.line
        character = element.range.start.character
        state = self.client.get_state_at_pos(filepath, line, character, timeout=timeout, retry=retry)

        steps = []
        initial_goals = self.client.goals(state)
        for tactic in proof_steps:
            dependencies = self.extract_dependencies(state, tactic, timeout=timeout, retry=retry)
            state = self.client.run(state, tactic, timeout=timeout, retry=retry)
            goals = self.client.goals(state, timeout=timeout, retry=retry)
            step = Step(tactic, goals, dependencies)
            steps.append(step)
        
        assert not goals, "Proof incomplete"
        return Theorem(steps, initial_goals, element)

    @staticmethod
    def _name_range_key(name: str, range: Range, offset=0):
        line_window = f'{range.start.line-offset}-{range.end.line-offset}' if range.start.line != range.end.line else f'{range.start.line-offset}'
        char_window = f'{range.start.character}-{range.end.character}' if range.start.character != range.end.character else f'{range.start.character}'
        return f'{name}, line: {line_window}, char: {char_window}'

    @staticmethod
    def _elem_key(element: Element, offset=0) -> str:
        return RocqParser._name_range_key(element.name, element.content_range, offset=0)

    @staticmethod
    def _toc_elem_key(toc_elem: TocElement, offset=0) -> str:
        return RocqParser._name_range_key(toc_elem.name.v, toc_elem.range, offset=0)

    def toc_to_element_tree(self, source: Source, proof_map: Dict[str, Range], toc_element: TocElement) -> Union[Element, Notation]:
        """
        Convert a TocElement node (and its TocElement children recursively)
        into an Element tree using `merge_toc_element` for node conversion.
        """
        pos_end = toc_element.range.end
        toc_key = RocqParser._toc_elem_key(toc_element)
        if toc_key in proof_map:
            pos_end = proof_map[toc_key].end
        state = self.client.get_state_at_pos(source.path, pos_end.line, pos_end.character)
        # if self.client.goals(state):
        #     state = self.client.run(state, 'Admitted.')
        is_notation = toc_element.detail=='Notation'
        if is_notation:
            return Notation(toc_element.name, toc_element.range, source.path, source.logical_path)
        
        print(toc_element)
        print(toc_key)
        print(proof_map.keys())
        print(toc_key in proof_map)
        print(pos_end)
        element = self.extract_element(state, toc_element.name.v, is_notation=is_notation)
        element.range = toc_element.range
        element.kind = toc_element.detail

        if toc_element.children:
            element.children = [self.toc_to_element_tree(source, child) for child in toc_element.children]

        return element

    def extract_toc(self, source: Source, timeout: int=60, retry=1) -> List[List[Union[Element,Notation]]]:
        """Read the table of contents for a source file."""
        result = []       
        toc = self.client.toc(source.path, timeout=timeout, retry=retry)
        proof_map = {RocqParser._elem_key(element, offset=1): r for element, r, _ in self.extract_proofs(source)}
        for _, toc_elements in toc:
            if not toc_elements:
                continue
            elements = [self.toc_to_element_tree(source, proof_map, toc_element) for toc_element in toc_elements]
            result.append(elements)
        return elements

    # def _extract_one_feedback(self, state: State):
    #     """Fetch the raw feedback string from a petanque state."""
    #     assert len(state.feedback) == 1 and state.feedback[0][0] == 3, f"Impossible to extract feedback from {state.feedback[0]}"
    #     return state.feedback[0][1]

    # def _parse_loadpath(self, feedback: str):
    #     """Parse `Print LoadPath` output into a module root map."""
    #     pattern = re.compile(r"^([A-Za-z0-9_]+)\s+(\S+)$", re.MULTILINE)
    #     roots = {}
    #     for name, path in pattern.findall(feedback):
    #         # Keep only the first (top-level) occurrence
    #         if name not in roots:
    #             roots[name] = path
    #     return roots
    
    # def _parse_locate(self, feedback: str):
    #     """Read module names from `Locate` output."""
    #     return re.findall(r"^Module\s+([A-Za-z0-9_.]+)", feedback, re.MULTILINE)

    # def extract_dependencies(self, source: Source, thms: List[Element]) -> Tuple[List[str], List[str]]:
    #     """Collect load paths and module dependencies from a source."""
    #     """
    #     TODO: for the moment Pytanque requires to be in a proof to use cmd such as About.
    #     """
    #     pattern = re.compile(
    #         r"""
    #         ^\s*
    #         (?:From\s+\S+\s+)?      # optional 'From <library>'
    #         Require\s+
    #         (?:Import|Export)?\s*   # optional 'Import' or 'Export'
    #         (?P<mods>[^.;]+)        # capture module list
    #         [.;]                    # end of statement
    #         """,
    #         re.VERBOSE | re.MULTILINE,
    #     )

    #     source_content = source.content
       
    #     all_modules = []
    #     for m in pattern.finditer(source_content):
    #         line_no = source_content.count("\n", 0, m.start()) + 1 # TODO: use it to recover the first usable state to Locate modules.
    #         raw = m.group(1)
    #         modules = [x for x in re.split(r"[\s,]+", raw.strip()) if x]
    #         all_modules += modules
    
    #     with Pytanque("127.0.0.1", self.pet_port) as client:
    #         for _ in range(10):
    #             thm = random.choice(thms)
    #             try:
    #                 state = client.start(source.path, thm.name)
    #                 break
    #             except PetanqueError:
    #                 pass
    #         loadpath_state = client.run(state, 'Print LoadPath.')
    #         feedback = self._extract_one_feedback(loadpath_state)
    #         loadpath = self._parse_loadpath(feedback)
    #         dependencies = []
    #         for module in all_modules:
    #             locatemodule_state = client.run(state, f'Locate {module}.')
    #             feedback = self._extract_one_feedback(locatemodule_state)
    #             dependency = self._parse_locate(feedback)
    #             dependencies += dependency
    #     return loadpath, dependencies
