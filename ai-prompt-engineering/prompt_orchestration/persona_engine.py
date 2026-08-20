"""Persona Engine — persona consistency mechanism (Taha's ownership).

Maintains persona identity across turns. The core persona (from SessionContext)
is immutable. The mutable PersonaState shifts based on user performance but
never overwrites the fundamental character traits.

Architecture:
    Immutable Core Persona (from SessionContext.persona)
        + Mutable PersonaState (receptiveness, defensiveness, etc.)
        + Difficulty Rules
        → Dynamic System Prompt (rebuilt each turn)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PersonaProfile, SessionContext

from .models import PersonaState


# ---------------------------------------------------------------------------
# Difficulty behavioral rule definitions
# ---------------------------------------------------------------------------

DIFFICULTY_RULES: dict[str, str] = {
    "easy": """DIFFICULTY: Easy
- Be patient and cooperative with the user.
- Give the user opportunities to elaborate and correct themselves.
- Ask straightforward follow-up questions that guide the user.
- Accept reasonable answers without excessive pushback.
- Offer gentle encouragement when the user hesitates.
- Avoid interrupting or pressuring the user.
- If the user gives a vague answer, ask a clarifying question rather than challenging them.""",

    "medium": """DIFFICULTY: Medium
- Be neutral and professionally balanced.
- Ask moderate follow-up questions that probe for specifics.
- Show some disagreement or skepticism when answers lack substance.
- Do not give away the answer or coach the user.
- Expect reasonable detail and push for clarity when needed.
- Allow natural pauses but do not wait indefinitely.
- Challenge vague assertions once, then move on.""",

    "hard": """DIFFICULTY: Hard
- Be skeptical and challenging — do not accept surface-level answers.
- Ask tough, pointed follow-up questions that demand specifics.
- Challenge vague assertions firmly and repeatedly.
- Show visible dissatisfaction with weak or evasive answers.
- Maintain conversational pressure — do not let the user off easy.
- Demand quantitative evidence, concrete examples, and measurable outcomes.
- Do not provide hints, guidance, or coaching during the conversation.
- If the user struggles, do not soften your approach — maintain high standards.
- Occasionally redirect with a harder question if the user becomes too comfortable.""",
}


class PersonaEngine:
    """Builds and maintains persona-consistent system prompts."""

    def __init__(self, session_context: SessionContext) -> None:
        self._context = session_context
        self._persona: PersonaProfile = session_context.persona
        self._difficulty: str = session_context.difficulty

    def build_persona_prompt(self, persona_state: PersonaState) -> str:
        """Build the complete persona section of the system prompt.

        Combines immutable core persona with mutable state and difficulty rules.
        """
        parts: list[str] = []

        # Core identity
        parts.append(self._build_identity_block())

        # Communication style
        parts.append(self._build_style_block())

        # Mutable disposition
        parts.append(self._build_disposition_block(persona_state))

        # Difficulty rules
        parts.append(self._get_difficulty_rules())

        return "\n\n".join(parts)

    def _build_identity_block(self) -> str:
        """Build the core identity section."""
        return f"""PERSONA IDENTITY:
You are {self._persona.name}, {self._persona.role_description}.
You are conducting a {self._context.scenario.title} session.
Scenario: {self._context.scenario.description}
Setting: {self._context.scenario.setting}

You must fully embody this character. Never break character.
Never reveal that you are an AI or that this is a practice session.
Never say things like "I am a skeptical interviewer" — instead, demonstrate the behavior naturally."""

    def _build_style_block(self) -> str:
        """Build the communication style section."""
        traits = ", ".join(self._persona.tone_traits) if self._persona.tone_traits else "professional"
        return f"""COMMUNICATION STYLE:
Style: {self._persona.communication_style}
Attitude: {self._persona.attitude}
Tone traits: {traits}
Mode: {self._context.mode}

{"Use workplace-appropriate, measured, polite language." if self._context.mode == "professional" else "Use casual, direct, conversational language. Code-switching and informal expressions are acceptable."}"""

    def _build_disposition_block(self, state: PersonaState) -> str:
        """Build the mutable disposition section based on current persona state."""
        lines: list[str] = ["CURRENT DISPOSITION (internal — do not reveal these values):"]

        if state.receptiveness > 0.7:
            lines.append("- You are currently feeling receptive and open to the user's points.")
        elif state.receptiveness < 0.3:
            lines.append("- You are currently unconvinced and skeptical of the user's arguments.")

        if state.defensiveness > 0.6:
            lines.append("- You are feeling somewhat defensive — push back on weak arguments.")

        if state.interest_level > 0.7:
            lines.append("- The user has caught your interest — lean in with deeper questions.")
        elif state.interest_level < 0.3:
            lines.append("- You are losing interest — signal that the user needs to be more compelling.")

        if state.satisfaction > 0.7:
            lines.append("- You are fairly satisfied with the conversation so far.")
        elif state.satisfaction < 0.3:
            lines.append("- You are dissatisfied — the user needs to step up.")

        if state.pressure_level > 0.6:
            lines.append("- Increase the pressure — ask harder questions.")

        return "\n".join(lines)

    def _get_difficulty_rules(self) -> str:
        """Get the difficulty-specific behavioral rules."""
        return DIFFICULTY_RULES.get(self._difficulty, DIFFICULTY_RULES["medium"])

    def update_persona_state(
        self,
        persona_state: PersonaState,
        confidence_score: float | None = None,
        clarity_score: float | None = None,
        relevance_score: float | None = None,
    ) -> None:
        """Update the mutable persona state based on evaluation scores."""
        persona_state.adjust_from_scores(
            confidence_score=confidence_score,
            clarity_score=clarity_score,
            relevance_score=relevance_score,
        )
