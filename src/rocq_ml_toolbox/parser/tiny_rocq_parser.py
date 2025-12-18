"""Tiny Rocq parser that replays proofs and records dependencies."""

from typing import List, Optional, Tuple, Generator, Dict
import re

from pytanque.protocol import State

from .utils.ast import list_dependencies
from .utils.message import parse_about, parse_loadpath
from .utils.position import extract_subtext, move_position
from .utils.toc import merge_toc_element

from .parser import Step, Position, Range, Element, Source, Dependency, Theorem, ParserError
from ..inference.client import PetClient, ClientError



class RocqParser:
    """Interact with petanque to collect proof structure and metadata."""

    def __init__(self, client: PetClient):
        """Create a parser bound to a client."""
        super().__init__()
        self.client = client
        self.map_logical_physical = self._extract_loadpath()
        self.map_physical_logical = {v: k for k,v in self.map_logical_physical.items()}
    
    @staticmethod
    def _extract_blocks(content: str):
        return re.split(r'(?<=\.\s)', content)

    def extract_element(self, state: State, element: str) -> Optional[Element]:
        substate = self.client.run(state, f'About {element}.')
        if substate.feedback:
            result = substate.feedback[0][1]
            return parse_about(result, self.map_logical_physical, self.map_physical_logical)
        return None

    def extract_dependencies(self, state: State, tactic: str) -> List[Dependency]:
        ast = self.client.ast(state, tactic)
        if ast:
            constants = list_dependencies(ast)
        else:
            constants = []
        dependencies = []
        for constant in constants:
            element = self.extract_element(state, constant)
            dependencies.append(element)
        return dependencies

    def _extract_loadpath(self) -> Dict[str, str]:
        try:
            state = self.client.get_root_state('/tmp/init.v')
        except ClientError:
            raise ParserError('Missing /tmp/init.v file.')
        result = self.client.run(state, 'Print LoadPath.')
        return parse_loadpath(result.feedback[0][1])

    def extract_ast(self, source: Source) -> Generator[Tuple[int, int, dict], None, None]:
        buffer = ""
        state = self.client.get_root_state(source.path)

        blocks = RocqParser._extract_blocks(source.content)
        curr_pos = Position(0, 0)
        for block in blocks:
            try:
                state = self.client.get_state_at_pos(source.path, curr_pos.line-1, curr_pos.character)
            except ClientError:
                pass
            buffer += block
            try:
                ast = self.client.ast(state, buffer)
                yield (curr_pos.line, curr_pos.character, ast)
                curr_pos = move_position(source.content, curr_pos, len(buffer))
                buffer = ""
            except ClientError:
                pass
    
    def extract_proofs(self, source: Source) -> Generator[Tuple[Element, List[str]], None, None]:
        ast = self.extract_ast(source)

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
                if content[0] == 'VernacStartTheoremProof':
                    infos = content[2][0][0][0]
                    name = infos['v'][1]
                    kind = content[1][0]
                    thm_name = name
                    start = node_end_pos
                    range_statement = Range(node_start_pos, node_end_pos)
                elif content[0] == 'VernacEndProof':
                    end = node_end_pos
                    if not (start and end and thm_name):
                        print(f"Ignore {node_start_pos} in {source.path} (no VernacStartTheoremProof associated).")
                        continue
                    
                    range_proof = Range(start, end)

                    proof = extract_subtext(source.content, range_proof)
                    statement = extract_subtext(source.content, range_statement)
                    proof_steps = RocqParser._extract_blocks(proof)

                    state = self.client.get_state_at_pos(source.path, end.line, end.character)
                    element = self.extract_element(state, thm_name)
                    assert element.physical_path == source.path, f"Path mismatch between {element.physical_path} and {source.path}"
                    element.content = statement
                    element.kind = kind
                    element.range = range_statement

                    thm_name: Optional[str] = None
                    start: Optional[Position] = None
                    kind: Optional[str] = None
                    range_statement: Optional[Range] = None
                    yield (element, proof_steps)
            except (KeyError, TypeError) as e:
                pass

    def compute_theorem(self, element: Element, proof_steps: List[str]) -> Theorem:
        filepath = element.physical_path
        line = element.range.start.line
        character = element.range.start.character
        state = self.client.get_state_at_pos(filepath, line, character)

        steps = []
        initial_goals = self.client.goals(state)
        for tactic in proof_steps:
            dependencies = self.extract_dependencies(state, tactic)
            state = self.client.run(state, tactic)
            goals = self.client.goals(state)
            step = Step(tactic, goals, dependencies)
            steps.append(step)
        
        assert not goals, "Proof incomplete"
        return Theorem(steps, initial_goals, element)

    def extract_toc(self, source: Source, timeout: int=120) -> List[Element]:
        """Read the table of contents for a source file."""
        elements = []       
        toc = self.client.toc(source.path, timeout=120)
        for name, toc_elements in toc:
            for toc_element in toc_elements:
                pos_start = element.range.start
                element_name = element.name.v
                state = self.client.get_state_at_pos(source.path, pos_start.line, pos_start.character)

                element = self.extract_element(state, element_name)
                merge_toc_element(element, toc_element)
                exit()
                # for subelement in self._flatten_element(element):
                #     found = False
                #     for detail in ALL_DETAILS:
                #         if subelement.detail in detail:
                #             found = True
                #             break
                #     if not found:
                #         print(subelement.detail)
        elements = toc
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

    def __call__(self, theorem: Element, source: Source) -> List[Step]:
        """Extract the proof steps for a single theorem."""
        return self._extract_proof(theorem, source)
