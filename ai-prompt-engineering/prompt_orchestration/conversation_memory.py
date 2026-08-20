"""Conversation Memory — context management strategy (Taha's ownership).

Implements three-tier memory:
  1. Short-term: Recent N turns in full
  2. Structured: Key facts extracted from conversation
  3. Long-term: Summary of older turns to prevent context overflow

This keeps the LLM prompt within a token budget while preserving
conversational continuity.
"""

from __future__ import annotations

import logging

from .models import ConversationMessage, ConversationState, StructuredMemory

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation context for prompt construction."""

    def __init__(
        self,
        max_recent_turns: int = 6,
        max_turns_before_summary: int = 10,
    ) -> None:
        self._max_recent = max_recent_turns
        self._max_before_summary = max_turns_before_summary

    def get_context_window(
        self, state: ConversationState
    ) -> dict[str, object]:
        """Build the context window for prompt construction.

        Returns:
            dict with keys:
                - recent_messages: list[ConversationMessage]
                - structured_memory: StructuredMemory
                - history_summary: str | None (summarized older turns)
        """
        messages = state.conversation_history

        # If history is short enough, use everything
        if len(messages) <= self._max_recent:
            return {
                "recent_messages": messages,
                "structured_memory": state.structured_memory,
                "history_summary": state.history_summary,
            }

        # Otherwise, keep only the most recent turns
        recent = messages[-self._max_recent:]

        # If we've exceeded the summary threshold and don't have a summary yet,
        # generate one from the older messages
        if len(messages) > self._max_before_summary and state.history_summary is None:
            older = messages[:-self._max_recent]
            state.history_summary = self._summarize_messages(older)
            logger.info(
                "Summarized %d older messages for session %s",
                len(older),
                state.session_id,
            )

        return {
            "recent_messages": recent,
            "structured_memory": state.structured_memory,
            "history_summary": state.history_summary,
        }

    def extract_facts(
        self,
        state: ConversationState,
        user_message: str,
        ai_response: str,
    ) -> None:
        """Extract key facts from the latest exchange into structured memory.

        Uses simple heuristic extraction (no LLM call needed for MVP).
        """
        memory = state.structured_memory

        # Extract claims — sentences with first-person assertions
        claim_indicators = [
            "i led", "i managed", "i built", "i created", "i delivered",
            "i increased", "i reduced", "i achieved", "my team", "i have",
            "years of experience", "i worked", "i developed", "i designed",
            "i implemented", "i improved", "i saved", "percent", "%",
        ]
        lower_msg = user_message.lower()
        for indicator in claim_indicators:
            if indicator in lower_msg:
                # Extract the sentence containing the indicator
                for sentence in user_message.split("."):
                    sentence = sentence.strip()
                    if indicator in sentence.lower() and sentence and len(sentence) > 10:
                        if sentence not in memory.user_claims:
                            memory.user_claims.append(sentence)
                            if len(memory.user_claims) > 15:
                                memory.user_claims.pop(0)
                        break

        # Track topics discussed from AI questions
        question_markers = ["?"]
        for marker in question_markers:
            if marker in ai_response:
                # Extract the question as a topic
                for sentence in ai_response.split("?"):
                    sentence = sentence.strip()
                    if len(sentence) > 15:
                        topic = sentence.split(".")[-1].strip()
                        if topic and topic not in memory.key_topics_discussed:
                            memory.key_topics_discussed.append(topic)
                            if len(memory.key_topics_discussed) > 10:
                                memory.key_topics_discussed.pop(0)
                        break

    def update_summary(self, state: ConversationState) -> None:
        """Force-update the history summary if history has grown too long."""
        messages = state.conversation_history
        if len(messages) > self._max_before_summary:
            older = messages[:-self._max_recent]
            state.history_summary = self._summarize_messages(older)

    @staticmethod
    def _summarize_messages(messages: list[ConversationMessage]) -> str:
        """Create a condensed summary of older messages.

        For MVP, this uses simple concatenation with truncation.
        Could be replaced with an LLM-based summarization call later.
        """
        parts: list[str] = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "AI"
            # Truncate long messages
            content = msg.content[:120] + "..." if len(msg.content) > 120 else msg.content
            parts.append(f"[Turn {msg.turn_id}] {role_label}: {content}")

        summary = "\n".join(parts)
        # Cap the summary length
        if len(summary) > 1500:
            summary = summary[:1500] + "\n[...earlier conversation truncated]"

        return summary

    @staticmethod
    def format_messages_for_prompt(messages: list[ConversationMessage]) -> str:
        """Format messages for inclusion in the LLM prompt."""
        lines: list[str] = []
        for msg in messages:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"You: {msg.content}")
        return "\n".join(lines)

    @staticmethod
    def format_memory_for_prompt(memory: StructuredMemory) -> str:
        """Format structured memory for inclusion in the LLM prompt."""
        parts: list[str] = []
        if memory.user_claims:
            parts.append("Key claims made by the user:")
            for claim in memory.user_claims[-5:]:
                parts.append(f"  - {claim}")
        if memory.key_topics_discussed:
            parts.append("Topics already discussed:")
            for topic in memory.key_topics_discussed[-5:]:
                parts.append(f"  - {topic}")
        if memory.unresolved_topics:
            parts.append("Unresolved topics to follow up on:")
            for topic in memory.unresolved_topics[-3:]:
                parts.append(f"  - {topic}")
        return "\n".join(parts)
