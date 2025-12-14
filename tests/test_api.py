# tests/test_api.py
import pytest
import requests
from inference_server.client import ClientError, PetClient

@pytest.mark.api
def test_health_check(server_url):
    resp = requests.get(f"{server_url}/health")
    assert resp.status_code == 200
    assert resp.text == "OK"

@pytest.mark.api
def test_login_creates_session(server_url):
    resp = requests.get(f"{server_url}/login")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0

@pytest.mark.api
def test_invalid_start_thm(server_url):
    """Test that bad inputs return correct error codes."""
    client = PetClient(server_url)
    with pytest.raises(ClientError) as excinfo:
        client.start_thm(filepath="", line=0, character=0)
    
    assert excinfo.value.code in [400, 404, 500]

@pytest.mark.api
def test_session_state_persistence(server_url):
    """Ensure we can retrieve session data."""
    client = PetClient(server_url)
    session_data = client.get_session()
    assert session_data['session_id'] == client.session_id