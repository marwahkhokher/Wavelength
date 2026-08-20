"""Qwen Metrics, Evaluation & Coaching Page module (Ahmed's ownership)."""

from .deep_evaluator import QwenDeepEvaluator
from .qwen_client import LiveQwenEvaluationClient
from .scorecard_generator import generate_session_scorecard
from .coaching_engine import CoachingEngine

__all__ = ["QwenDeepEvaluator", "LiveQwenEvaluationClient", "generate_session_scorecard", "CoachingEngine"]
