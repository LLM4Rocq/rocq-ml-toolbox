"""Tiny Rocq parser that replays proofs and records dependencies."""

from typing import List, Optional, Tuple, Generator, Dict, Union
import re

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

    def extract_element(self, state: State, element: str, timeout: int=30, retry=1, is_notation=False) -> Tuple[Optional[Element], bool]:
        try:
            if is_notation:
                substate = self.client.run(state, f'About "{element}".', timeout=timeout, retry=retry)
            else:
                substate = self.client.run(state, f'About {element}.', timeout=timeout, retry=retry)
            if substate.feedback:
                result = substate.feedback[0][1]
                return parse_about(result, self.map_logical_physical, self.map_physical_logical)
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
        except ClientError:
            raise ParserError('Missing /tmp/init.v file.')
        result = self.client.run(state, 'Print LoadPath.', timeout=timeout, retry=retry)
        return parse_loadpath(result.feedback[0][1])

    def extract_ast(self, source: Source, retry=1) -> Generator[Tuple[str, dict, State, Range], None, None]:
        fleche_document = self.client.get_document(source.path)
        for ranged_span in fleche_document.spans[:-1]:
            text = ranged_span.span
            r = ranged_span.range
            ast = self.client.ast_at_pos(source.path, r.start.line, r.start.character)
            state = self.client.get_state_at_pos(source.path, r.end.line, r.end.character)
            yield text, state, r, ast

    def extract_toc(self, source: Source, timeout=60, retry=1, solve_deps=False) -> Generator[Theorem, None, None]:
        fleche_document = self.client.get_document(source.path, timeout=timeout)
        steps = []
        old_state = self.client.get_root_state(source.path, timeout=timeout, retry=retry)
        state = old_state
        initial_goals = None
        for ranged_span in fleche_document.spans[:-1]:
            text = ranged_span.span
            r = ranged_span.range
            ast = self.client.ast_at_pos(source.path, r.start.line, r.start.character, timeout=timeout, retry=retry)
            old_state = state
            state = self.client.get_state_at_pos(source.path, r.end.line, r.end.character, timeout=timeout, retry=retry)
            content = ast['v']['expr'][1]
            ast_kind = content[0]

            # print(ast_kind)
            # print(text)
            hb_detected = False
            if ast_kind == 'VernacExtend':
                if 2 <= len(content) and isinstance(content[1], dict) and 'ext_entry' in content[1]:
                    if content[1]['ext_entry'].startswith('ElpiHB'):
                        keyword = content[1]['ext_entry'].replace('ElpiHB', 'HB')
                        hb_detected = True
                        if keyword != 'HB.export':
                            try:
                                ast = self.client.ast(state, text.split(keyword)[1])
                                content = ast['v']['expr'][1]
                                ast_kind = content[0]
                            except ClientError:
                                pass
            if not hb_detected and ast_kind in ['VernacExtend', 'VernacBullet', 'VernacEndProof', 'VernacProof', 'VernacSubproof', 'VernacEndSubproof']:
                goals = self.client.goals(state)
                dependencies = []
                if solve_deps:
                    dependencies = self.extract_dependencies(state, text, timeout=timeout, retry=retry)
                step = Step(text, goals, dependencies)
                steps.append(step)
            else:
                if steps and name and initial_goals:
                    element, is_local = self.extract_element(old_state, name, timeout=timeout, retry=retry)
                    if element and is_local:
                        yield Theorem(steps, initial_goals, element)
                        steps = []
                    else:
                        print(f"Warning: Ignore element before {text} at {r} ({ast_kind})")
            if ast_kind == 'VernacStartTheoremProof':
                infos = content[2][0][0][0]
                name = infos['v'][1]
                initial_goals = self.client.goals(state)
            elif ast_kind == 'VernacNotation':
                infos = content[2]
                name = infos['ntn_decl_string']['v']
                notation = Notation(name, r, source.path, source.logical_path)
                yield notation
            elif ast_kind == 'VernacDefinition':
                infos = content[2][0]
                name = infos['v'][1][1]
                initial_goals = self.client.goals(state)
                element, is_local = self.extract_element(state, name, timeout=timeout, retry=retry)
                if element and is_local:
                    yield element
            elif ast_kind == 'VernacInstance':
                infos = content[1][0]
                name = infos['v'][1][1]
                initial_goals = self.client.goals(state)
                element, is_local = self.extract_element(state, name, timeout=timeout, retry=retry)
                if element and is_local:
                    yield element
            elif ast_kind == 'VernacFixpoint':
                infos = content[2][1][0]
                name = infos['fname']['v'][1]
                element, is_local = self.extract_element(state, name, timeout=timeout, retry=retry)
                if element and is_local:
                    yield element
            elif ast_kind == 'VernacInductive':
                infos = content[2][0][0][0][1][0]
                name = infos['v'][1]
                children = []
                for subinfos in content[2][0][0]:
                    if isinstance(subinfos, list) and subinfos and subinfos[0] == 'Constructors':
                        subinfos = subinfos[1]
                        for constructor in subinfos:
                            constructor = constructor[1][0]
                            constructor_name = constructor['v'][1]
                            child, is_local = self.extract_element(state, constructor_name, timeout=timeout, retry=retry)
                            if child and is_local:
                                child.kind = "Constructor"
                                children.append(child)
                    if isinstance(subinfos, list) and subinfos and subinfos[0] == 'RecordDecl':
                        subinfos = subinfos[2]
                        for field in subinfos:
                            field = field[0][1]
                            field_name = field['v'][1][1]
                            child, is_local = self.extract_element(state, field_name, timeout=timeout, retry=retry)
                            if child and is_local:
                                child.kind = "Record"
                                children.append(child)
                            
                element, is_local = self.extract_element(state, name, timeout=timeout, retry=retry)
                if element and is_local:
                    element.children = children
                    yield element

    # def toc_to_element_tree(self, source: Source, toc_element: TocElement) -> Union[Element, Notation, None]:
    #     """
    #     Convert a TocElement node (and its TocElement children recursively)
    #     into an Element tree using `merge_toc_element` for node conversion.
    #     """
    #     pos_end = toc_element.range.end
    #     state = self.client.get_state_at_pos(source.path, pos_end.line, pos_end.character)
    #     is_notation = toc_element.detail=='Notation'
    #     if is_notation:
    #         return Notation(toc_element.name, toc_element.range, source.path, source.logical_path)
    #     element, _ = self.extract_element(state, toc_element.name.v, is_notation=is_notation)
    #     if element:
    #         element.range = toc_element.range
    #         element.kind = toc_element.detail

    #         if toc_element.children:
    #             element.children = [self.toc_to_element_tree(source, child) for child in toc_element.children]

    #         return element
    #     return None

    # def extract_toc(self, source: Source, timeout: int=60, retry=1) -> Generator[Union[Theorem, Element, Notation], None, None]:
    #     """Read the table of contents for a source file."""   
    #     toc = self.client.toc(source.path, timeout=timeout, retry=retry)
    #     for _, toc_elements in toc:
    #         if not toc_elements:
    #             continue

    #         for toc_element in toc_elements:
    #             element_tree = self.toc_to_element_tree(source, toc_element)
    #             if element_tree:
    #                 print(toc_element)
    #             #     yield element_tree
    #     exit()
    #     for thm in self.extract_proofs(source, timeout=timeout, retry=retry):
    #         yield thm