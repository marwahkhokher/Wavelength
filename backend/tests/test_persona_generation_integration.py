from __future__ import annotations

import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATABASE_URL", "sqlite:///./persona-generation-test.db")
os.environ.setdefault("JWT_SECRET_KEY", "persona-generation-test-secret")
langchain_openai = types.ModuleType("langchain_openai")
langchain_openai.ChatOpenAI = object
sys.modules.setdefault("langchain_openai", langchain_openai)

from app.models.enums import Mode
from app.services import persona_generation


class FakeLLM:
    def invoke(self, messages):  # noqa: ANN001
        return type(
            "Response",
            (),
            {
                "content": (
                    '{"name":"Alex","age_range":"unknown",'
                    '"role_or_relationship":"Direct manager",'
                    '"background":"No background information provided.",'
                    '"personality_traits":["direct"],'
                    '"communication_style":["asks for concrete examples"],'
                    '"goals_in_conversation":["understand the request"],'
                    '"potential_triggers":["vague answers"],'
                    '"tone":{"speech_register":"measured and polite",'
                    '"deflection_style":"diplomatic",'
                    '"example_phrase":"Let me consider that."},'
                    '"mode":"professional","difficulty":"hard",'
                    '"initial_state":{"receptiveness":0.5,"patience":0.5,"trust":0.5},'
                    '"generated_from_scenario":"wrong",'
                    '"generated_from_persona_description":"wrong",'
                    '"is_finalized":true}'
                )
            },
        )()


def test_backend_generation_uses_rich_prompt_and_forces_unfinalized_profile(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(persona_generation, "get_llm", lambda temperature: FakeLLM())

    profile = persona_generation.generate_persona_profile(
        "I need to ask my manager for a raise.",
        "Direct manager who asks for evidence.",
        Mode.PROFESSIONAL,
        "hard",
    )

    assert profile["generated_from_scenario"] == "I need to ask my manager for a raise."
    assert profile["generated_from_persona_description"] == "Direct manager who asks for evidence."
    assert profile["mode"] == "professional"
    assert profile["difficulty"] == "hard"
    assert profile["is_finalized"] is False
    assert profile["tone"]["speech_register"] == "measured and polite"