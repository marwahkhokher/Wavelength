"""Unit tests for SessionState, SessionManager, and ConversationMemory (Taha's ownership)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "voice-tech-infra" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompt_orchestration.conversation_memory import ConversationMemory
from prompt_orchestration.models import (
    ConversationMessage,
    ConversationState,
    SessionState,
    StructuredMemory,
)
from prompt_orchestration.session_manager import SessionManager


def test_session_manager_lifecycle():
    manager = SessionManager()
    session = manager.create_session(session_id="s_123", user_id="u_abc")

    assert session.session_id == "s_123"
    assert session.state == SessionState.ACTIVE

    # Retrieve
    retrieved = manager.get_session("s_123")
    assert retrieved is not None
    assert retrieved.user_id == "u_abc"

    # End
    ended = manager.end_session("s_123")
    assert ended is not None
    assert ended.state == SessionState.COMPLETED

    # Delete
    assert manager.remove_session("s_123") is True
    assert manager.get_session("s_123") is None


def test_conversation_memory_windowing_and_claims():
    mem = ConversationMemory(max_recent_turns=4, max_turns_before_summary=6)
    state = ConversationState(session_id="s_mem", user_id="u_mem")

    # Add 8 exchanges (16 messages)
    for i in range(1, 9):
        state.add_message("user", f"I led project {i} and increased revenue by {i*10} percent.")
        state.add_message("assistant", f"How did you measure the impact of project {i}?")

    # Context window should contain only the latest 4 messages and a summary of older ones
    context = mem.get_context_window(state)
    recent = context["recent_messages"]
    assert len(recent) == 4
    assert "project 8" in recent[-2].content

    # Structured memory extraction
    mem.extract_facts(
        state=state,
        user_message="I managed a team of five engineers and achieved a 40 percent speedup.",
        ai_response="That's impressive. What was the biggest hurdle?",
    )
    assert len(state.structured_memory.user_claims) > 0
    assert any("managed a team" in c for c in state.structured_memory.user_claims)


def test_conversation_memory_prompt_formatting():
    messages = [
        ConversationMessage(role="user", content="Hello", turn_id=1),
        ConversationMessage(role="assistant", content="Welcome! Let's begin.", turn_id=1),
    ]
    formatted = ConversationMemory.format_messages_for_prompt(messages)
    assert "User: Hello" in formatted
    assert "You: Welcome! Let's begin." in formatted

    struct_mem = StructuredMemory(
        user_claims=["Led migration to microservices", "Increased throughput by 50%"],
        key_topics_discussed=["System architecture"],
    )
    mem_formatted = ConversationMemory.format_memory_for_prompt(struct_mem)
    assert "microservices" in mem_formatted
    assert "System architecture" in mem_formatted


if __name__ == "__main__":
    test_session_manager_lifecycle()
    test_conversation_memory_windowing_and_claims()
    test_conversation_memory_prompt_formatting()
    print("All session state & memory unit tests passed!")
