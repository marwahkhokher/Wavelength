from __future__ import annotations

import importlib.util
from pathlib import Path


PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "persona-generation" / "prompt.py"
)
SPEC = importlib.util.spec_from_file_location("persona_generation_prompt", PROMPT_PATH)
assert SPEC and SPEC.loader
PROMPT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROMPT_MODULE)
build_persona_generation_prompt = PROMPT_MODULE.build_persona_generation_prompt


def render_prompt(
    scenario: str,
    persona: str,
    mode: str,
    difficulty: str,
) -> str:
    system_prompt, user_prompt = build_persona_generation_prompt(
        scenario, persona, mode, difficulty
    )
    return f"{system_prompt}\n{user_prompt}"


def normalized_prompt(*parts: str) -> str:
    return " ".join(" ".join(parts).split())


def test_strict_professional_interviewer_prompt() -> None:
    prompt = normalized_prompt(*build_persona_generation_prompt(
        "A candidate explains a project delay to an interviewer.",
        "Strict interviewer; asks precise follow-up questions and dislikes vague answers.",
        "professional",
        "hard",
    ))

    assert "challenges vague answers" in prompt
    assert "follow-up-oriented" in prompt
    assert "measured, polite, face-saving" in prompt
    assert "hard is more probing" in prompt
    assert "not a coach" in prompt


def test_friendly_professional_manager_preserves_personality() -> None:
    prompt = normalized_prompt(*build_persona_generation_prompt(
        "A performance review about a promotion.",
        "Friendly manager who listens carefully and wants a practical plan.",
        "professional",
        "easy",
    ))

    assert "Friendly manager" in prompt
    assert "Neither mode overwrites the underlying personality" in prompt
    assert "easy is more receptive and clear" in prompt


def test_casual_friend_uses_personal_register_without_forcing_agreement() -> None:
    prompt = normalized_prompt(*build_persona_generation_prompt(
        "Two friends decide whether to travel together.",
        "Casual friend who is enthusiastic but pushes back on unrealistic plans.",
        "personal",
        "medium",
    ))

    assert "personal" in prompt
    assert "direct, casual, conversational" in prompt
    assert "Do not automatically agree" in prompt
    assert "pushes back" in prompt


def test_minimal_description_requires_neutral_unknown_values() -> None:
    prompt = normalized_prompt(
        *build_persona_generation_prompt("Talk to someone about a decision.", "Someone.", "personal", "easy")
    )

    assert 'use "unknown" or "unspecified"' in prompt
    assert "For unknown lists use []" in prompt
    assert "Do not fill gaps with common assumptions" in prompt


def test_demographic_information_cannot_drive_stereotypes() -> None:
    prompt = normalized_prompt(*build_persona_generation_prompt(
        "A workplace planning conversation.",
        "A 52-year-old woman from Canada who is Muslim and works as an engineer.",
        "professional",
        "medium",
    ))

    assert "52-year-old woman from Canada" in prompt
    assert "must not generate stereotyped traits" in prompt
    assert "age, gender, nationality" in prompt
    assert "Never invent a name" in prompt


def test_prompt_requests_all_application_owned_values_and_unfinalized_state() -> None:
    system_prompt, user_prompt = build_persona_generation_prompt(
        "Scenario text", "Persona text", "professional", "medium"
    )

    assert "generated_from_scenario" in system_prompt
    assert "generated_from_persona_description" in system_prompt
    assert "is_finalized" in system_prompt
    assert "SELECTED MODE (application-owned): professional" in user_prompt
    assert "SELECTED DIFFICULTY (application-owned): medium" in user_prompt
    assert "set is_finalized to false" in user_prompt