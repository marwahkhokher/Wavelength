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

### Live Qwen evaluation

By default, the evaluator uses its deterministic local scorer. To enable real
Qwen scoring, copy `ai-prompt-engineering/.env.example` to the repository-root
`.env` and restart the service. Two options:

- **Local, free (default):** install [Ollama](https://ollama.com), run
  `ollama pull qwen3.5:4b`, and leave the `.env.example` defaults as-is
  (`QWEN_BASE_URL=http://localhost:11434/v1`). No API key is needed - Ollama
  exposes an OpenAI-compatible endpoint that `LiveQwenEvaluationClient`
  (`qwen_evaluation/qwen_client.py`) talks to directly.
- **Hosted, paid:** set `DASHSCOPE_API_KEY` and the region-specific
  `QWEN_BASE_URL` for Alibaba Model Studio instead (see the commented-out
  block in `.env.example`).

`QwenDeepEvaluator` picks the live client automatically whenever
`QWEN_BASE_URL` is set. If the provider is unavailable or returns invalid
JSON, the service logs the failure and uses the deterministic scorer instead.

## Folders
- `persona-generation/` — prompts and logic for turning subcategory + free text into a persona config
- `dynamic-engine/` — turn-by-turn evaluation and persona state shifting
- `scoring-feedback/` — end-of-session rubric and evidence-based feedback generation
- `growth-tracking/` — cross-session tagging and pattern summarization
