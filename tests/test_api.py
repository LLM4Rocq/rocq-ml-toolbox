# tests/test_api.py
import pytest
import requests
from inference_server.client import ClientError

@pytest.mark.api
def test_health_check(gunicorn_server):
    resp = requests.get(f"{gunicorn_server}/health")
    assert resp.status_code == 200
    assert resp.text == "OK"

@pytest.mark.api
def test_login_creates_session(gunicorn_server):
    resp = requests.get(f"{gunicorn_server}/login")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0

@pytest.mark.api
def test_invalid_start_thm(client):
    """Test that bad inputs return correct error codes."""
    with pytest.raises(ClientError) as excinfo:
        client.start_thm(filepath="", line=0, character=0)
    
    assert excinfo.value.code in [400, 404, 500]

@pytest.mark.api
def test_session_state_persistence(client):
    """Ensure we can retrieve session data."""
    session_data = client.get_session()
    assert session_data['session_id'] == client.session_id