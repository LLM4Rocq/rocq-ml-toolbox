"""Tiny Rocq parser that replays proofs and records dependencies."""

from typing import List, Optional, Tuple
import re
from copy import deepcopy
import random

from pytanque import Pytanque, State, PetanqueError

from src.parser.parser import AbstractParser, Step, Position, Range, Element, Source, update_statement, Dependency, ProofNotFound

def read_keyword(keyword: str, l: list, result: list[str]) -> list[str]:
    """Collect AST nodes tagged with the given keyword."""

    if isinstance(l, list):
        if len(l) >= 3 and l[0] == keyword:
            result.append((l[1], l[2]))
            l = l[3:]

        for el in l:
            result = read_keyword(keyword, el, result)

    elif isinstance(l, dict):
        for el in l.values():
            result = read_keyword(keyword, el, result)

    return result

def list_dependencies(ast: dict) -> list[str]:
    """Extract clean dependency names from an AST."""
    expr = ast["v"]["expr"]
    raw_dependencies = read_keyword("Ser_Qualid", expr, [])

    dependencies = []
    for dir_path, name in raw_dependencies:
        dependencies.append(".".join(map(lambda w: w[1], dir_path[1] + [name])))

    return [dependency for i, dependency in enumerate(dependencies) if not dependency in dependencies[:i]]

class TinyRocqParser(AbstractParser):
    """Interact with petanque to collect proof structure and metadata."""

    def __init__(self, pet_port, timeout=30):
        """Create a parser bound to a pet-server port."""
        super().__init__()
        self.pet_port = pet_port
        self.timeout = timeout

    def _extract_proof_steps(self, theorem: Element, source: Source):
        """Split a proof script into tactic steps."""
        idx_offset = 0
        subsource_lines = source.content_lines[theorem.range.end.line:]
        subsource_lines[0] = subsource_lines[0][theorem.range.end.character:]
        for line in subsource_lines:
            idx_offset += 1
            if 'Qed.' in line:
                break
            if 'Abort.' in line or 'Admitted.' in line:
                raise ProofNotFound
        proof_block = "\n".join(subsource_lines[:idx_offset])
        proof_attempt = re.split(r'(?<=[^\.]\.)\s+', proof_block)
        result = []
        for line in proof_attempt:
            line = line.strip()
            for symbol in ['-', '+', '*']:
                if line.startswith(symbol):
                    result.append(symbol)
                    line = line[len(symbol):].strip()
            result.append(line)
        return result
    
    def _parse_about(self, result: str) -> Optional[Dependency]:
        """Turn `About` feedback into a dependency record."""
        name = result.split(' :')[0]
        if "Hypothesis of the goal context." in result:
            return Dependency(origin="", name=name, range=None, kind='hypothesis')
        if 'Declared in' in result:
            pattern = (
                r'Declared in\s+(?:File\s+"(?P<file>[^"]+)"|library (?P<lib>[^,]+)), '
                r'line (?P<line>\d+(?:-\d+)?), characters (?P<char>\d+(?:-\d+)?)'
            )

            match = re.search(pattern, result)
            assert match, f"Issue when parsing position information from {result}"
            origin = match.group(1) or match.group(2)
            line_part = match.group(3)
            char_part = match.group(4)

            line_start, line_end = (map(int, line_part.split('-')) if '-' in line_part else (int(line_part), int(line_part)))
            char_start, char_end = (map(int, char_part.split('-')) if '-' in char_part else (int(char_part), int(char_part)))

            r = Range(
                start=Position(line_start, char_start),
                end=Position(line_end, char_end)
            )
            return Dependency(origin=origin, name=name, range=r, kind='premise')
        return None
        
    def _extract_proof(self, theorem: Element, source: Source):
        """Replay a proof and capture the states plus dependencies."""
        proof_attempt = self._extract_proof_steps(theorem, source)
        proof_check = []
        with Pytanque("127.0.0.1", self.pet_port) as client:
            state = client.start(source.path, theorem.name)
            goals_out = client.goals(state)
            for line in proof_attempt:
                state_in = deepcopy(goals_out)
                ast = client.ast(state, line)
                if ast:
                    constants = list_dependencies(ast)
                else:
                    constants = []
                dependencies = []
                for constant in constants:
                    substate = client.run(state, f'About {constant}.')
                    if substate.feedback:
                        result = substate.feedback[0][1]
                        dependancy = self._parse_about(result)
                        if dependancy:
                            dependencies.append(dependancy)
                state = client.run(state, line, timeout=self.timeout)
                goals = client.goals(state)
                step = Step(step=line, state_in=state_in, state_out=goals, dependencies=dependencies)
                proof_check.append(step)
            assert not goals, "Proof incomplete"
        return proof_check

    def extract_toc(self, source: Source) -> List[Element]:
        """Read the table of contents for a source file."""
        elements = []
        with Pytanque("127.0.0.1", self.pet_port) as client:
            for name, details in client.toc(source.path):
                if details[-1]['detail'] in ['Lemma', 'Theorem']:
                    theorem = Element.from_dict(details[-1] | {"origin": str(source.path), "name": name, "statement": "statement"})
                    update_statement(theorem, source)
                    elements.append(theorem)
        return elements
    
    def _extract_one_feedback(self, state: State):
        """Fetch the raw feedback string from a petanque state."""
        assert len(state.feedback) == 1 and state.feedback[0][0] == 3, f"Impossible to extract feedback from {state.feedback[0]}"
        return state.feedback[0][1]

    def _parse_loadpath(self, feedback: str):
        """Parse `Print LoadPath` output into a module root map."""
        pattern = re.compile(r"^([A-Za-z0-9_]+)\s+(\S+)$", re.MULTILINE)
        roots = {}
        for name, path in pattern.findall(feedback):
            # Keep only the first (top-level) occurrence
            if name not in roots:
                roots[name] = path
        return roots
    
    def _parse_locate(self, feedback: str):
        """Read module names from `Locate` output."""
        return re.findall(r"^Module\s+([A-Za-z0-9_.]+)", feedback, re.MULTILINE)

    def extract_dependencies(self, source: Source, thms: List[Element]) -> Tuple[List[str], List[str]]:
        """Collect load paths and module dependencies from a source."""
        """
        TODO: for the moment Pytanque requires to be in a proof to use cmd such as About.
        """
        pattern = re.compile(
            r"""
            ^\s*
            (?:From\s+\S+\s+)?      # optional 'From <library>'
            Require\s+
            (?:Import|Export)?\s*   # optional 'Import' or 'Export'
            (?P<mods>[^.;]+)        # capture module list
            [.;]                    # end of statement
            """,
            re.VERBOSE | re.MULTILINE,
        )

        source_content = source.content
       
        all_modules = []
        for m in pattern.finditer(source_content):
            line_no = source_content.count("\n", 0, m.start()) + 1 # TODO: use it to recover the first usable state to Locate modules.
            raw = m.group(1)
            modules = [x for x in re.split(r"[\s,]+", raw.strip()) if x]
            all_modules += modules
    
        with Pytanque("127.0.0.1", self.pet_port) as client:
            for _ in range(10):
                thm = random.choice(thms)
                try:
                    state = client.start(source.path, thm.name)
                    break
                except PetanqueError:
                    pass
            loadpath_state = client.run(state, 'Print LoadPath.')
            feedback = self._extract_one_feedback(loadpath_state)
            loadpath = self._parse_loadpath(feedback)
            dependencies = []
            for module in all_modules:
                locatemodule_state = client.run(state, f'Locate {module}.')
                feedback = self._extract_one_feedback(locatemodule_state)
                dependency = self._parse_locate(feedback)
                dependencies += dependency
        return loadpath, dependencies

    def __call__(self, theorem: Element, source: Source) -> List[Step]:
        """Extract the proof steps for a single theorem."""
        return self._extract_proof(theorem, source)
