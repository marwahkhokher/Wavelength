# Voice/Tech Infrastructure

Owns real-time voice: speech-to-text, text-to-speech, latency and turn-taking, and session state during a live conversation.

FastAPI + websockets service that runs the live voice loop: streaming STT (Deepgram) -> turn-taking (with barge-in) -> AI service call -> streaming TTS (ElevenLabs), on top of a session-state layer that survives reconnects and duplicate tabs.

## Layout

```
src/wavelength_voice/
  config.py           Settings (env vars / .env)
  app_state.py         Process-wide singletons shared across connections
  main.py               FastAPI app factory + entrypoint
  ai_service/           Contract shared with the AI/Prompt Engineering team
    contracts.py           AITurnRequest / AITurnResponse pydantic models
    client.py               AIServiceClient interface, MockAIServiceClient, HTTPAIServiceClient
  voice_pipeline/        (maps to the `voice-pipeline/` concept in the team README)
    stt.py                  STTStream interface + DeepgramSTTStream
    tts.py                   TTSStream interface + ElevenLabsTTSStream
    turn_taking.py            TurnTakingController - turn state machine + barge-in
  session_state/          (maps to `session-state/`)
    models.py                 Session, ConversationTurn, SessionStatus
    manager.py                 SessionManager - reconnect grace window + duplicate-tab takeover
  ws/
    router.py                  /ws/session/{session_id} websocket endpoint
    orchestrator.py             Per-connection glue: STT -> turn-taking -> AI -> TTS

latency-tests/            (maps to `latency-tests/`)
  turn_latency_benchmark.py  Standalone script measuring time-to-first-audio-chunk

tests/                     pytest suite (see "Tests" below)
```

## Setup

```bash
cd voice-tech-infra
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e . -r requirements.txt
cp .env.example .env          # fill in DEEPGRAM_API_KEY / ELEVENLABS_API_KEY for real STT/TTS
```

## Run

```bash
uvicorn wavelength_voice.main:app --reload --app-dir src
```

- `GET /healthz` - liveness check.
- `WS /ws/session/{session_id}?user_id=...` - the voice loop. Binary frames in/out are audio; JSON text frames carry control/status events (`connected`, `agent_reply_text`, `agent_speech_ended`, `session_taken_over`, `session_ended`, `error`).

By default `USE_MOCK_AI_SERVICE=true`, so the service runs end-to-end against `MockAIServiceClient` without any AI/Prompt Engineering dependency. Flip it to `false` and set `AI_SERVICE_BASE_URL` once their real `/v1/turn` endpoint exists - it just needs to accept/return the shapes in `ai_service/contracts.py`.

## Design notes

**Session resilience** (`session_state/manager.py`): `SessionManager.connect()` is the single entry point for a websocket attaching to a session.
- New `session_id` -> a session is created.
- Reconnect within `SESSION_RECONNECT_GRACE_SECONDS` of a disconnect -> same session, conversation history intact.
- Grace window elapsed -> session is `EXPIRED`; further connects are rejected (client should start a new session).
- A second connection to an already-`ACTIVE` session (duplicate tab) -> the new connection wins ("last tab wins"); the manager reports `evicted_connection_id` and the router closes that old websocket with a `session_taken_over` message. A disconnect from an already-evicted connection is a no-op, so a slow-closing old tab can't clobber the new one's state.

This logic is pure and synchronous-feeling (an injectable clock, no I/O) specifically so it's cheap to unit test without spinning up real websockets - see `tests/test_session_manager.py`.

**Turn-taking + barge-in** (`voice_pipeline/turn_taking.py`): a 4-state machine (`WAITING_FOR_USER` -> `USER_SPEAKING` -> `PROCESSING` -> `AGENT_SPEAKING` -> back to `WAITING_FOR_USER`). A `user_speech_started` event arriving during `PROCESSING` or `AGENT_SPEAKING` is a barge-in: it interrupts the turn, fires `on_barge_in`, and the orchestrator (`ws/orchestrator.py`) cancels the in-flight AI-service call and/or TTS stream in response. Illegal sequencing (e.g. an agent response arriving before the user finished talking) raises `InvalidTurnTransition` rather than silently corrupting state.

**Mocked AI-service client** (`ai_service/`): `contracts.py` defines the wire contract (`AITurnRequest` in, `AITurnResponse` out); `MockAIServiceClient` implements it in-process (scripted replies, simulated latency, optional `end_after_turns`) so the whole voice pipeline is buildable/testable before the AI/Prompt Engineering team's real service exists. `HTTPAIServiceClient` is the drop-in real implementation - same contract, POSTs to `{AI_SERVICE_BASE_URL}/v1/turn`.

## Tests

```bash
pytest
```

38 tests covering:
- `test_session_manager.py` - reconnect within/past the grace window, duplicate-tab takeover, stale-connection disconnects being ignored, concurrent connect races, session expiry/end.
- `test_turn_taking.py` - the full turn cycle, barge-in from both `PROCESSING` and `AGENT_SPEAKING`, invalid-transition errors.
- `test_ai_service_client.py` - the request/response contract (validation, mock client behavior, HTTP client success/error paths).
- `test_orchestrator.py` - end-to-end wiring with fake STT/TTS streams: a full turn, a barge-in cancelling TTS mid-stream, and a slow AI response being dropped when the user talks over it.

## Not yet wired up

- Real credentials/live testing against Deepgram and ElevenLabs (the wrapper protocol implementations in `stt.py`/`tts.py` are written against their documented streaming APIs but untested against live traffic in this environment).
- Persisting session state anywhere durable - `SessionManager` is in-memory per-process, fine for a single instance but won't survive a restart or work across multiple server instances without an external store (e.g. Redis) backing it.
