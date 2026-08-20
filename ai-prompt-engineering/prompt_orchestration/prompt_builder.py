"""Modular Prompt Builder (Taha's ownership).

Constructs structured, persona-consistent prompts for the Conversation LLM.
Combines:
  1. Role & Scenario rules
  2. Persona identity & communication style
  3. Dynamic disposition & Difficulty rules
  4. Structured memory (extracted facts & claims)
  5. Recent conversation history
  6. Latest user message
  7. Turn analysis (optional / background coaching hints)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add project root and voice-tech-infra root to sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root.parent / "voice-tech-infra" / "src"))

from wavelength_voice.ai_service.contracts import PerTurnEvaluation, SessionContext

from .conversation_memory import ConversationMemory
from .models import ConversationState
from .persona_engine import PersonaEngine


class PromptBuilder:
    """Modular prompt builder generating roleplay-consistent prompts."""

    def __init__(self, session_context: SessionContext, memory_manager: ConversationMemory | None = None) -> None:
        self.session_context = session_context
        self.persona_engine = PersonaEngine(session_context)
        self.memory_manager = memory_manager or ConversationMemory()

    def build_system_prompt(self, state: ConversationState) -> str:
        """Builds the comprehensive system instruction for the LLM."""
        persona_block = self.persona_engine.build_persona_prompt(state.persona_state)

        rules = """GENERAL CONVERSATION RULES:
1. Stay strictly in character at all times.
2. DO NOT act as an AI assistant, coach, or meta-evaluator during the conversation.
3. DO NOT praise the user's communication style or give tips like "good use of STAR method" or "your tone was confident".
4. React realistically as the character in the given scenario.
5. Keep your responses natural, conversational, and direct (typically 1-3 sentences per turn for natural spoken voice dialogue).
6. Ask follow-up questions that challenge, probe, or advance the scenario conversation.
7. Do not fabricate facts that contradict the scenario."""

        return f"{persona_block}\n\n{rules}"

    def build_turn_prompt(
        self,
        state: ConversationState,
        latest_user_text: str,
        qwen_eval: PerTurnEvaluation | None = None,
    ) -> str:
        """Builds the prompt context payload including history, memory, and latest user text."""
        context = self.memory_manager.get_context_window(state)
        recent_messages = context["recent_messages"]
        structured_mem = context["structured_memory"]
        history_summary = context["history_summary"]

        sections: list[str] = []

        # 1. Summary of older conversation if available
        if history_summary:
            sections.append(f"### SUMMARY OF EARLIER CONVERSATION:\n{history_summary}")

        # 2. Structured memory
        memory_str = self.memory_manager.format_memory_for_prompt(structured_mem)
        if memory_str:
            sections.append(f"### STRUCTURED MEMORY / KNOWN USER CLAIMS:\n{memory_str}")

        # 3. Recent conversation history
        history_str = self.memory_manager.format_messages_for_prompt(recent_messages)
        if history_str:
            sections.append(f"### RECENT CONVERSATION:\n{history_str}")
        else:
            sections.append("### RECENT CONVERSATION:\n(This is the beginning of the session. You are delivering the opening turn.)")

        # 4. Background Evaluation / Subconscious perception (if provided)
        if qwen_eval:
            # Internal context only - the persona feels this intuitively rather than reciting metrics
            perception = []
            if qwen_eval.scores.confidence < 50:
                perception.append("User sounds noticeably hesitant and lacking conviction.")
            elif qwen_eval.scores.confidence > 80:
                perception.append("User sounds confident and poised.")

            if qwen_eval.scores.clarity < 50:
                perception.append("User's explanation is somewhat vague or difficult to follow.")

            if perception:
                sections.append(f"### YOUR INTERNAL IMPRESSION OF USER'S LAST UTTERANCE:\n- " + "\n- ".join(perception))

        # 5. Latest user utterance
        sections.append(f"User's latest message:\n\"{latest_user_text}\"")
        sections.append(
            "Respond naturally as your character in spoken voice dialogue. "
            "Keep it concise (1-3 sentences), stay in persona, and react directly to what the user just said."
        )

        return "\n\n".join(sections)

    def build_opening_prompt(self, state: ConversationState) -> str:
        """Builds the prompt for the AI to deliver the initial scenario greeting / opening question."""
        sections = [
            "### SESSION START:",
            f"You are opening the session for the scenario: \"{self.session_context.scenario.title}\".",
            f"Scenario details: {self.session_context.scenario.description}",
            "Generate a strong, natural opening line or initial question to kick off the interaction in character.",
            "Keep it concise (1-2 sentences) and inviting or demanding depending on your persona and difficulty level."
        ]
        return "\n\n".join(sections)
