"""Unit tests for PromptBuilder and PersonaEngine (Taha's ownership)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_orchestration.conversation_memory import ConversationMemory
from prompt_orchestration.models import ConversationState, PersonaState
from prompt_orchestration.persona_engine import DIFFICULTY_RULES, PersonaEngine
from prompt_orchestration.prompt_builder import PromptBuilder
from prompt_orchestration.providers.mock_providers import MockSessionContextProvider


def test_persona_engine_easy_mode():
    provider = MockSessionContextProvider()
    ctx = provider.build_context(
        session_id="test_easy",
        user_id="u1",
        mode="professional",
        difficulty="easy",
        persona_name="Sarah",
        persona_role="Lead Architect",
    )
    engine = PersonaEngine(ctx)
    state = PersonaState(receptiveness=0.8, satisfaction=0.8)
    prompt = engine.build_persona_prompt(state)

    assert "Sarah" in prompt
    assert "Lead Architect" in prompt
    assert "DIFFICULTY: Easy" in prompt
    assert "Be patient and cooperative" in prompt
    assert "receptive and open" in prompt


def test_persona_engine_hard_mode():
    provider = MockSessionContextProvider()
    ctx = provider.build_context(
        session_id="test_hard",
        user_id="u2",
        mode="personal",
        difficulty="hard",
        persona_name="Zubair",
        persona_role="Co-founder",
    )
    engine = PersonaEngine(ctx)
    state = PersonaState(receptiveness=0.2, defensiveness=0.8, pressure_level=0.8)
    prompt = engine.build_persona_prompt(state)

    assert "Zubair" in prompt
    assert "DIFFICULTY: Hard" in prompt
    assert "skeptical and challenging" in prompt
    assert "MODE: Personal" in prompt or "casual, direct" in prompt
    assert "Increase the pressure" in prompt


def test_persona_state_shift_on_scores():
    state = PersonaState(receptiveness=0.5, satisfaction=0.5, pressure_level=0.3)
    # High confidence & clarity
    state.adjust_from_scores(confidence_score=85.0, clarity_score=90.0, relevance_score=90.0)
    assert state.receptiveness > 0.5
    assert state.satisfaction > 0.5

    # Low confidence triggers pressure increase
    state.adjust_from_scores(confidence_score=30.0, clarity_score=40.0, relevance_score=40.0)
    assert state.pressure_level > 0.3


def test_prompt_builder_turn_prompt():
    provider = MockSessionContextProvider()
    ctx = provider.build_context(
        session_id="test_pb",
        user_id="u3",
        mode="professional",
        scenario_title="System Design",
        scenario_description="Architecting high-throughput pipeline",
        persona_name="David",
        persona_role="VP",
    )
    builder = PromptBuilder(ctx)
    state = ConversationState(session_id="test_pb", user_id="u3")
    state.add_message("user", "I led the development of our distributed cache.")
    state.add_message("assistant", "What was the cache invalidation strategy?")

    sys_prompt = builder.build_system_prompt(state)
    assert "David" in sys_prompt
    assert "GENERAL CONVERSATION RULES" in sys_prompt

    turn_prompt = builder.build_turn_prompt(
        state=state,
        latest_user_text="We used write-through caching with Redis clusters.",
    )
    assert "Redis clusters" in turn_prompt
    assert "distributed cache" in turn_prompt
    assert "Respond naturally" in turn_prompt


def test_prompt_builder_opening():
    provider = MockSessionContextProvider()
    ctx = provider.build_context(
        session_id="test_open",
        user_id="u4",
        mode="professional",
        scenario_title="Salary Negotiation",
        scenario_description="Annual raise",
        persona_name="Helen",
        persona_role="Director",
    )
    builder = PromptBuilder(ctx)
    state = ConversationState(session_id="test_open", user_id="u4")
    opening_prompt = builder.build_opening_prompt(state)

    assert "Salary Negotiation" in opening_prompt
    assert "SESSION START" in opening_prompt


if __name__ == "__main__":
    test_persona_engine_easy_mode()
    test_persona_engine_hard_mode()
    test_persona_state_shift_on_scores()
    test_prompt_builder_turn_prompt()
    test_prompt_builder_opening()
    print("All prompt builder unit tests passed!")
