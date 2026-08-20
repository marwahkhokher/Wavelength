# AI/Prompt Engineering

Owns the "brain" of the app: persona generation from free text, the dynamic receptiveness/resistance engine, scoring and feedback generation, and cross-session growth pattern detection.

## AI service

The deployable FastAPI boundary is [`service.py`](service.py). It accepts the
Voice/Tech contract at `POST /v1/turn` and returns the next conversation reply
plus the per-turn evaluation.

Run it from the repository root:

```powershell
.venv\\Scripts\\python -m uvicorn service:app --app-dir ai-prompt-engineering --port 9000
```

Existing Voice/Tech callers can continue sending the original transcript-only
request. For full Qwen evaluation, include the optional `session_context`,
`stt_result`, and `tone_result` objects defined in
`voice-tech-infra/src/wavelength_voice/ai_service/contracts.py`.

## Folders
- `persona-generation/` — prompts and logic for turning subcategory + free text into a persona config
- `dynamic-engine/` — turn-by-turn evaluation and persona state shifting
- `scoring-feedback/` — end-of-session rubric and evidence-based feedback generation
- `growth-tracking/` — cross-session tagging and pattern summarization
