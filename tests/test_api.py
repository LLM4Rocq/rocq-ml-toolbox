import os
from typing import Tuple, List

import pytest
import requests

from pytanque.protocol import (
    Goal,
    Inspect,
    InspectGoals,
    InspectPhysical,
    GoalsResponse
)

from rocq_ml_toolbox.inference.client import StateExtended


def _find_line_char(path_abs: str, needle: str) -> Tuple[int, int]:
    with open(path_abs, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            j = line.find(needle)
            if j != -1:
                return i, j  # 0-based
    raise AssertionError(f"Needle not found: {needle}")


def _pick_thm() -> Tuple[str, List[str]]:
    return "Z_of_of_Z", [
            ('cbv [of_Z to_Z]; destruct z as [|z|z];'
            ' cbn; try reflexivity; rewrite Raw.to_N_of_pos;'
            ' cbn; reflexivity.'
            ),
            'Qed.'
        ]


def _extract_statement() -> str:
    return ('Lemma Z_of_of_Z (z : Z)'
 ' : to_Z (of_Z z)'
 ' = z.')

@pytest.fixture(scope="session")
def server_file(stdlib_path):
    file_path = os.path.join(stdlib_path, "Strings/BinaryString.v")
    return file_path

@pytest.fixture(scope="session")
def needle_pos(server_file):
    return _find_line_char(server_file, "cbv [of_Z to_Z]; destruct z as [|z|z];")


@pytest.fixture(scope="session")
def lemma_pos(server_file):
    return _find_line_char(server_file, "Lemma Z_of_of_Z (z : Z)")


@pytest.fixture(scope="session")
def statement_state(client, server_file, lemma_pos):
    line, character = lemma_pos
    st = client.get_state_at_pos(server_file, line, character)
    return st

@pytest.fixture(scope="session")
def started_state(client, server_file):
    thm,proof= _pick_thm()
    st = client.start(file=server_file, thm=thm, timeout=180)
    return thm, proof, st

@pytest.mark.api
def test_get_state_at_pos_endpoint(client, server_file, needle_pos):
    line0, ch0 = needle_pos
    st = client.get_state_at_pos(filepath=server_file, line=line0, character=ch0, timeout=180)
    assert isinstance(st, StateExtended)

@pytest.mark.api
def test_run_endpoint(client, started_state):
    _, proof, st0 = started_state
    for step in proof:
        st0 = client.run(st0, step, timeout=120)
    goals = client.goals(st0)
    assert goals==[]

@pytest.mark.api
def test_goals_endpoint(client, started_state):
    _, _, st0 = started_state
    goals = client.goals(st0, pretty=True, timeout=30)
    assert isinstance(goals, list)
    for goal in goals:
        assert isinstance(goal, Goal)

@pytest.mark.api
def test_complete_goals_endpoint(client, started_state):
    _, _, st0 = started_state
    goals = client.complete_goals(st0, pretty=True, timeout=30)
    assert isinstance(goals, GoalsResponse) or not goals

@pytest.mark.api
def test_premises_endpoint(client, started_state):
    _, _, st0 = started_state
    premises = client.premises(st0, timeout=30)
    assert isinstance(premises, list)

@pytest.mark.api
def test_state_equal_endpoint(client, started_state):
    _, _, st0 = started_state
    inspect_physical = Inspect(InspectPhysical())
    assert client.state_equal(st0, st0, inspect_physical, timeout=30) is True

    st1 = client.run(st0, 'idtac.')
    inspect_goals = Inspect(InspectGoals())

    assert client.state_equal(st0, st1, inspect_goals, timeout=30) is True
    assert client.state_equal(st0, st1, inspect_physical, timeout=30) is False

@pytest.mark.api
def test_state_hash_endpoint(client, started_state):
    _, _, st0 = started_state
    h = client.state_hash(st0, timeout=30)
    assert isinstance(h, (str, int))

@pytest.mark.api
def test_toc(client, server_file):
    toc = client.toc(server_file)
    assert isinstance(toc, list)

@pytest.mark.api
def test_ast_endpoint(client, started_state):
    _, _, st0 = started_state
    ast = client.ast(st0, text="test_tac.", timeout=30)
    assert ast is not None

@pytest.mark.api
def test_ast_at_pos(client, server_file, lemma_pos):
    line0, ch0 = lemma_pos
    ast = client.ast_at_pos(file=server_file, line=line0, character=ch0, timeout=60)
    assert ast is not None

@pytest.mark.api
def test_get_root_state_endpoint(client, server_file):
    st = client.get_root_state(file=server_file, timeout=180)
    assert isinstance(st, StateExtended) and st

@pytest.mark.api
def test_list_notations_in_statement_endpoint(client, statement_state):
    st0 = statement_state
    statement = _extract_statement()
    notations = client.list_notations_in_statement(st0, statement=statement, timeout=60)
    assert isinstance(notations, list)

@pytest.mark.api
def test_start_endpoint(client, server_file):
    thm,_ = _pick_thm()
    st = client.start(file=server_file, thm=thm, timeout=180)
    assert isinstance(st, StateExtended) and st

# @pytest.mark.api
# def test_query(client, server_file):
#     thm,_ = _pick_thm()
#     params = StartParams(server_file, thm, "")
#     resp = client.query(params, timeout=60)
#     assert isinstance(resp, Response)

@pytest.mark.api
def test_get_document_endpoint(client, server_file):
    client.get_document(server_file)

@pytest.mark.api
def test_get_ast_endpoint(client, server_file):
    client.get_ast(server_file, force_dump=True)

@pytest.mark.api
def test_get_session_endpoint(client):
    sess = client.get_session()
    assert sess["session_id"] == client.session_id
