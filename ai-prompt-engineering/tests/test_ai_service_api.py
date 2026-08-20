"""API tests for the FastAPI AI Service (Taha's ownership)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from prompt_orchestration.ai_service_app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_session_start_and_get():
    start_payload = {
        "session_id": "api_test_sess",
        "user_id": "usr_99",
        "mode": "professional",
        "scenario_title": "Quarterly Review",
        "scenario_description": "Reviewing team deliverables",
        "persona_name": "Marcus",
        "persona_role": "VP Product",
        "difficulty": "medium",
        "duration_seconds": 300,
    }
    resp = client.post("/v1/session/start", json=start_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "api_test_sess"
    assert data["persona_name"] == "Marcus"
    assert len(data["opening_reply"]) > 0

    # GET session state
    get_resp = client.get("/v1/session/api_test_sess")
    assert get_resp.status_code == 200
    sess_data = get_resp.json()
    assert sess_data["session_id"] == "api_test_sess"
    assert sess_data["history_count"] == 1


def test_turn_wire_contract():
    turn_payload = {
        "session_id": "api_test_sess",
        "user_id": "usr_99",
        "persona": {
            "persona_id": "p_marcus",
            "name": "Marcus",
            "scenario_prompt": "VP Product checking quarterly metrics",
            "difficulty": "medium",
            "traits": ["direct", "demanding"],
        },
        "transcript": "We successfully shipped 3 major features on time with zero critical bugs.",
        "turn_number": 1,
    }
    resp = client.post("/v1/turn", json=turn_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reply_text" in data
    assert data["reply_text"] is not None
    assert len(data["reply_text"]) > 0
    assert data["end_session"] is False


def test_interrupt_and_end_session():
    # Interrupt
    int_resp = client.post("/v1/session/api_test_sess/interrupt")
    assert int_resp.status_code == 200
    assert int_resp.json()["status"] == "interrupted"

    # End
    end_resp = client.post("/v1/session/api_test_sess/end")
    assert end_resp.status_code == 200
    assert end_resp.json()["status"] == "completed"


if __name__ == "__main__":
    test_health()
    test_session_start_and_get()
    test_turn_wire_contract()
    test_interrupt_and_end_session()
    print("All AI service API tests passed!")
