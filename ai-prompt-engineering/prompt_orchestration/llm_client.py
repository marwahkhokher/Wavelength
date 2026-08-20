"""Conversation LLM Client (Taha's ownership).

Supports Google Gemini via direct HTTP REST (httpx) and SDK, with automatic fallback
to a deterministic in-process MockLLMClient for testing and offline development.
All secrets and model parameters are sourced from config / environment variables.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
from typing import AsyncGenerator

import httpx

from .config import get_ai_settings

logger = logging.getLogger(__name__)


class BaseLLMClient(abc.ABC):
    """Abstract interface for Conversation LLM clients."""

    @abc.abstractmethod
    async def generate_response(self, system_instruction: str, prompt: str) -> str:
        """Generate response text given a system instruction and user prompt."""
        raise NotImplementedError

    @abc.abstractmethod
    async def stream_response(self, system_instruction: str, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response tokens given a system instruction and user prompt."""
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Deterministic, offline LLM mock for unit testing and development.

    Produces context-aware, realistic persona responses based on prompt keywords
    and scenario context without requiring an active network connection or API key.
    """

    def __init__(self, latency_seconds: float = 0.05) -> None:
        self.latency_seconds = latency_seconds

    async def generate_response(self, system_instruction: str, prompt: str) -> str:
        if self.latency_seconds > 0:
            await asyncio.sleep(self.latency_seconds)

        is_personal = "MODE: Personal" in system_instruction or "casual and candid" in system_instruction
        is_hard = "DIFFICULTY: Hard" in system_instruction or "skeptical" in system_instruction.lower()

        # Check for opening prompt
        if "SESSION START" in prompt or "opening line" in prompt:
            if is_personal:
                return "Hey! Thanks for catching up with me. What was on your mind?"
            return "Good morning. Thank you for coming in today. Let's start with your background and what brings you here."

        prompt_lower = prompt.lower()

        # Context-sensitive heuristics
        if "salary" in prompt_lower or "raise" in prompt_lower or "compensation" in prompt_lower:
            if is_personal:
                return "Yaar, everyone wants more money! But tell me honestly, why right now?"
            if is_hard:
                return "Budget constraints are tight this quarter. What specific quantifiable metrics justify an out-of-cycle compensation adjustment?"
            return "I understand your request. Could you walk me through your key achievements over the last review cycle?"

        if "velocity" in prompt_lower or "metric" in prompt_lower or "team" in prompt_lower:
            if is_personal:
                return "That sounds cool, but how did the rest of the team feel about the changes?"
            if is_hard:
                return "Increased velocity is good, but did code quality or sprint completion suffer as a result?"
            return "That's an interesting point regarding team velocity. What specific processes helped achieve that?"

        if "leadership" in prompt_lower or "managed" in prompt_lower or "led" in prompt_lower:
            if is_personal:
                return "Leading people isn't easy! What was the hardest part for you?"
            return "Can you share a specific instance where your leadership directly resolved a conflict or technical roadblock?"

        # Default fallback
        if is_personal:
            return "Hmm, that makes sense. But what do you think we should do next?"
        if is_hard:
            return "I see what you are saying, but what evidence do you have that this approach will scale?"
        return "Thank you for explaining that. Could you elaborate on how you plan to measure success moving forward?"

    async def stream_response(self, system_instruction: str, prompt: str) -> AsyncGenerator[str, None]:
        full_text = await self.generate_response(system_instruction, prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)


class GeminiLLMClient(BaseLLMClient):
    """Production client using Google Gemini REST API."""

    _CANDIDATE_MODELS = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_ai_settings()
        self.api_key = api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name or settings.conversation_model or "gemini-2.5-flash"
        self.temperature = temperature if temperature is not None else settings.conversation_temperature
        self._mock_fallback = MockLLMClient()

    async def generate_response(self, system_instruction: str, prompt: str) -> str:
        if not self.api_key:
            return await self._mock_fallback.generate_response(system_instruction, prompt)

        # Try configured model, and if 404, try alternative candidate models
        models_to_try = [self.model_name] + [m for m in self._CANDIDATE_MODELS if m != self.model_name]

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            payload = {
                "system_instruction": {
                    "parts": [{"text": system_instruction}]
                },
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": 512,
                }
            }
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                self.model_name = model  # Remember working model
                                return parts[0]["text"].strip()
                    elif resp.status_code == 404:
                        logger.warning("Gemini model %s returned 404, trying next candidate...", model)
                        continue
                    else:
                        logger.warning("Gemini API error %d: %s", resp.status_code, resp.text)
            except Exception as e:
                logger.warning("Gemini request exception for model %s: %s", model, e)
                continue

        logger.warning("All Gemini model requests failed, using MockLLMClient fallback")
        return await self._mock_fallback.generate_response(system_instruction, prompt)

    async def stream_response(self, system_instruction: str, prompt: str) -> AsyncGenerator[str, None]:
        full_text = await self.generate_response(system_instruction, prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)


def get_llm_client(force_mock: bool = False) -> BaseLLMClient:
    """Factory creating GeminiLLMClient if API key exists, otherwise MockLLMClient."""
    settings = get_ai_settings()
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")

    if force_mock or not api_key:
        logger.info("Using MockLLMClient (API key absent or mock requested)")
        return MockLLMClient()

    return GeminiLLMClient(api_key=api_key, model_name=settings.conversation_model)
