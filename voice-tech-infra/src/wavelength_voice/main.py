"""FastAPI application entrypoint.

Run with: ``uvicorn wavelength_voice.main:app --reload`` (from ``src/``), or
``python -m wavelength_voice.main`` for a plain dev server.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from wavelength_voice.ai_service.client import HTTPAIServiceClient, MockAIServiceClient
from wavelength_voice.app_state import AppState
from wavelength_voice.config import get_settings
from wavelength_voice.session_state.manager import SessionManager
from wavelength_voice.ws.router import router as ws_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    session_manager = SessionManager(
        reconnect_grace_seconds=settings.session_reconnect_grace_seconds
    )
    ai_client = (
        MockAIServiceClient()
        if settings.use_mock_ai_service
        else HTTPAIServiceClient(
            base_url=settings.ai_service_base_url,
            timeout_seconds=settings.ai_service_timeout_seconds,
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sweeper_task = asyncio.create_task(
            session_manager.run_expiry_sweeper(
                settings.session_expiry_sweep_interval_seconds
            )
        )
        yield
        sweeper_task.cancel()

    app = FastAPI(title="Wavelength Voice/Tech Infrastructure", lifespan=lifespan)
    app.state.wavelength = AppState(
        settings=settings, session_manager=session_manager, ai_client=ai_client
    )
    app.include_router(ws_router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
