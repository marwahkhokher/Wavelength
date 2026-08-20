"""Contract tests for the deployable AI FastAPI service."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import app


def test_turn_endpoint_returns_a_response_and_rich_evaluation() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/turn",
        json={
            "session_id": "session-1",
            "user_id": "user-1",
            "persona": {
                "persona_id": "manager-1",
                "name": "Alex",
                "scenario_prompt": "Discuss a salary increase after a strong performance year.",
            },
            "transcript": "First, I automated the release process and reduced delivery delays.",
            "turn_number": 1,
            "stt_result": {
                "transcript": "First, I automated the release process and reduced delivery delays.",
                "total_words": 10,
                "filler_word_count": 0,
                "utterance_duration_sec": 4.5,
            },
            "tone_result": {
                "primary_emotion": "confident",
                "pause_metrics": {
                    "total_pause_duration_sec": 0.1,
                    "pause_count": 1,
                    "max_pause_sec": 0.1,
                },
                "speech_rate_wpm": 135,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reply_text"]
    assert body["evaluation"]["scores"]["confidence"] > 80
    assert body["evaluation"]["suggested_conversation_followup_direction"]


def test_turn_endpoint_keeps_the_original_voice_request_compatible() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/turn",
        json={
            "session_id": "session-2",
            "user_id": "user-2",
            "persona": {
                "persona_id": "manager-2",
                "name": "Sam",
                "scenario_prompt": "Practice a workplace conversation.",
            },
            "transcript": "I would like to explain my approach.",
            "turn_number": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["evaluation"] is not None
