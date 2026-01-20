import os
from typing import List

import pytest

from rocq_ml_toolbox.parser.rocq_parser import Source, Theorem, VernacElement

@pytest.mark.parser
def test_toc(parser, stdlib_filepaths):
    for filepath in stdlib_filepaths[:1]:
        source = Source.from_local_path(filepath)
        parser.extract_toc(source)

@pytest.mark.parser
def test_extract_proofs(parser, stdlib_filepaths):
    for filepath in stdlib_filepaths[:1]:
        source = Source.from_local_path(filepath)
        parser.extract_proofs(source)

@pytest.mark.parser
def test_ast_one_to_one(client, stdlib_filepaths):
    for filepath in stdlib_filepaths[:1]:
        for element in client.get_ast(filepath):
            assert element == VernacElement.from_json(element.to_json())

@pytest.mark.parser
def test_theorem_one_to_one(parser, stdlib_filepaths):
    for filepath in stdlib_filepaths[:1]:
        source = Source.from_local_path(filepath)
        for proof in parser.extract_proofs(source):
            assert proof == Theorem.from_json(proof.to_json())
