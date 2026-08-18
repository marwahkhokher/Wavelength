"""Process-wide singletons shared across websocket connections."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import WebSocket

from wavelength_voice.ai_service.client import AIServiceClient
from wavelength_voice.config import Settings
from wavelength_voice.session_state.manager import SessionManager


@dataclass
class AppState:
    settings: Settings
    session_manager: SessionManager
    ai_client: AIServiceClient
    #: connection_id -> live websocket, so a duplicate-tab takeover can find
    #: and forcibly close the connection it just evicted.
    connections: dict[str, WebSocket] = field(default_factory=dict)
