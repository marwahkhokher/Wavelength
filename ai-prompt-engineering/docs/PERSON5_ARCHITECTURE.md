# Person 5 (Taha) — Architecture & Integration Documentation

**Role**: Senior AI Engineer and Real-Time Voice AI Architect  
**Ownership**: Person 5 (Main Conversation LLM + Modular Prompt Builder + ElevenLabs Streaming TTS + Real-Time Orchestration + Barge-In Interruption)  
**Package Locations**: `prompt_orchestration/` and `LLM_prompt/`  

---

## 1. System Architecture Overview

Person 5 sits downstream of all four AI teammates and interfaces directly with the Voice Infrastructure layer (`voice-tech-infra`):

```
                        [ Session Initialization ]
                                    │
                                    ▼
                          SessionContext (Armeen)
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │        Session State & Memory         │
                │        (SessionManager / Cache)       │
                └───────────────────┬───────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
Person 2: Areej             Person 3: Zaid                  Person 4: Ahmed
Whisper STTResult           Tone & EmotionResult            Qwen TurnAnalysis
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     ConversationOrchestrator  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         PromptBuilder         │
                    │  - System Prompt + Persona    │
                    │  - Scenario + Difficulty      │
                    │  - Memory + History + Turn    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │        Conversation LLM       │
                    │   (Gemini 2.0 Flash / Mock)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     ElevenLabs Streaming TTS  │
                    │     - Chunked WebSocket audio │
                    │     - Barge-in Cancellation   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                             Client / User
```

---

## 2. Input Schemas (Downstream Integration)

The orchestrator consumes contract types defined in `voice-tech-infra/src/wavelength_voice/ai_service/contracts.py`:

### A. SessionContext (Person 1 — Armeen)
```json
{
  "session_id": "sess_987654",
  "user_id": "usr_12345",
  "mode": "professional",
  "difficulty": "medium",
  "duration_seconds": 300,
  "scenario": {
    "title": "Salary Negotiation",
    "description": "Negotiating a 15% raise with a manager after a strong performance year.",
    "setting": "Annual performance review meeting room."
  },
  "persona": {
    "name": "David Miller",
    "role_description": "VP of Engineering, direct, polite, budget-conscious.",
    "communication_style": "Measured, professional, face-saving, avoids blunt refusals.",
    "attitude": "firm but receptive",
    "tone_traits": ["diplomatic", "structured", "face-saving"]
  }
}
```

### B. STTResult (Person 2 — Areej)
```json
{
  "transcript": "Um, I believe my contribution to the project, like, increased team velocity significantly.",
  "is_final": true,
  "total_words": 13,
  "filler_word_count": 2,
  "filler_words": [
    { "word": "um", "start_time": 0.12, "end_time": 0.45 },
    { "word": "like", "start_time": 1.80, "end_time": 2.05 }
  ],
  "utterance_duration_sec": 3.8
}
```

### C. ToneResult (Person 3 — Zaid)
```json
{
  "primary_emotion": "hesitant",
  "emotion_confidence_scores": { "hesitant": 0.78, "anxious": 0.15, "confident": 0.07 },
  "pitch_energy_variation": 0.42,
  "hesitation_score": 0.65,
  "pause_metrics": { "total_pause_duration_sec": 0.9, "pause_count": 2, "max_pause_sec": 0.6 },
  "speech_rate_wpm": 135.5,
  "silence_ratio": 0.23
}
```

### D. PerTurnEvaluation (Person 4 — Ahmed)
```json
{
  "turn_index": 1,
  "scores": {
    "clarity": 78.0,
    "empathy": 65.0,
    "filler_words_score": 70.0,
    "structure": 65.0,
    "relevance": 90.0,
    "confidence": 60.0,
    "overall_turn_score": 71.3
  },
  "coach_tip": "Pause 1 second before speaking to improve structure."
}
```

---

## 3. Output Schemas

### A. HTTP Wire Contract (`AITurnResponse` for `POST /v1/turn`)
```json
{
  "reply_text": "That's an interesting point regarding team velocity. What specific metrics supported that growth?",
  "persona_state": {
    "disposition": "Disposition: rec=0.55, sat=0.55",
    "difficulty": "medium"
  },
  "end_session": false,
  "feedback_hint": "Pause 1 second before speaking to improve structure.",
  "latency_ms": 65
}
```

### B. PromptLLMOutput (`prompt_llm.generate_next_turn()`)
```python
PromptLLMOutput(
    reply_text="Can you elaborate on how you measured that 25% increase?",
    end_session=False,
    persona_state_update="Disposition: rec=0.60, sat=0.55"
)
```

---

## 4. How to Run

### Run the FastAPI AI Service (port 9000):
```bash
python -m prompt_orchestration.ai_service_app
# or
python -m LLM_prompt.ai_service_app
```

### Run All Unit and Integration Tests:
```bash
python tests/test_prompt_builder.py
python tests/test_session_state.py
python tests/test_providers.py
python tests/test_conversation_orchestrator.py
python tests/test_ai_service_api.py
python tests/test_e2e_mock.py
python tests/test_pipeline.py
```

---

## 5. Required Environment Variables

Create `.env` based on `prompt_orchestration/.env.example`:

```env
# Gemini Conversation LLM
GEMINI_API_KEY=your_gemini_api_key_here
CONVERSATION_MODEL=gemini-2.0-flash
CONVERSATION_TEMPERATURE=0.7
CONVERSATION_MAX_TOKENS=512

# ElevenLabs Streaming TTS
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_MODEL_ID=eleven_turbo_v2_5
ELEVENLABS_OUTPUT_FORMAT=pcm_16000

# Provider Selection (mock | real)
PROVIDER_MODE=mock

# Server Configuration
AI_SERVICE_HOST=0.0.0.0
AI_SERVICE_PORT=9000
LOG_LEVEL=info
```

---

## 6. Provider Adapter Architecture (Mock vs Real)

The system uses an **Adapter/Provider pattern** so development continues even if a teammate's model is unavailable:

```
TranscriptProvider (ABC)
   ├── MockTranscriptProvider    (Simulates transcripts & fillers)
   └── WhisperTranscriptProvider (Wraps Person 2's WhisperSTTEngine)

ToneProvider (ABC)
   ├── MockToneProvider          (Simulates acoustics & emotion2vec)
   └── RealToneProvider          (Wraps Person 3's EmotionClassifier)

AnalysisProvider (ABC)
   ├── MockAnalysisProvider      (Simulates metric scoring)
   └── QwenAnalysisProvider      (Wraps Person 4's QwenDeepEvaluator)

SessionContextProvider (ABC)
   ├── MockSessionContextProvider (Simulates SessionContext)
   └── RealSessionContextProvider (Wraps Person 1's build_session_context)
```

### How to Switch Between Mock and Real:
Set in `.env`:
```env
PROVIDER_MODE=real   # Uses Person 1-4 code
# or
PROVIDER_MODE=mock   # Uses self-contained mock providers
```
Or pass directly in Python code:
```python
orchestrator = ConversationOrchestrator(use_mock_providers=False)
```

---

## 7. Modular Prompt Construction

The `PromptBuilder` constructs prompts dynamically:

```
┌────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT                                          │
│  - Persona Identity & Background                       │
│  - Communication Style (Professional vs Personal)       │
│  - Current Disposition (Receptiveness, Pressure)       │
│  - Difficulty Rules (Easy / Medium / Hard)             │
│  - Anti-coaching & character immersion constraints     │
└────────────────────────────────────────────────────────┘
                           +
┌────────────────────────────────────────────────────────┐
│ TURN PROMPT PAYLOAD                                    │
│  - Summary of older turns (if history > 10 turns)      │
│  - Extracted claims & facts (Structured Memory)        │
│  - Recent conversation turns (last 6 messages)         │
│  - Internal perception hint from Qwen score (optional) │
│  - Latest user message                                 │
└────────────────────────────────────────────────────────┘
```

---

## 8. Conversation Memory Strategy

A three-tier memory architecture minimizes token costs while preserving continuity:
1. **Short-Term Memory**: Keeps the last 6 messages in full verbatim history.
2. **Structured Session Memory**: Automatically extracts user claims (e.g. *"I led a team of 4"*, *"increased revenue by 30%"*) and key topics discussed.
3. **Long-Conversation Summary**: Summarizes turns 1..N-6 into a compact narrative paragraph when the conversation exceeds 10 turns.

---

## 9. Persona Consistency Mechanism

- **Core Identity is Immutable**: Persona role, core traits, and scenario rules from `SessionContext` never change during a session.
- **Disposition is Mutable**: The persona's `receptiveness`, `satisfaction`, and `pressure_level` dynamically adjust based on user performance (e.g. high confidence increases receptiveness; low confidence increases conversational pressure).
- **Immersion Enforcement**: The LLM is explicitly forbidden from giving coaching advice or saying "I am skeptical". It must demonstrate skepticism through direct probing questions.

---

## 10. ElevenLabs Streaming TTS & Barge-In Cancellation

1. **Streaming Audio**: Audio chunks are generated in real-time and yielded as an `AsyncGenerator[bytes, None]`.
2. **Barge-In (`Cut AI off`)**: When the user speaks while the AI is talking:
   - Voice infrastructure signals barge-in via `POST /v1/session/{id}/interrupt` or `orchestrator.trigger_barge_in()`.
   - The active TTS stream is instantly cancelled.
   - Any buffered audio chunks are discarded.
   - The state machine immediately transitions to `USER_SPEAKING`.

---

## 11. Testing & Validation

The test suite covers:
- **Unit Tests**: `test_prompt_builder.py`, `test_session_state.py`, `test_providers.py`
- **Integration Tests**: `test_conversation_orchestrator.py`, `test_ai_service_api.py`
- **End-to-End Simulation**: `test_e2e_mock.py`
- **Teammates Integration**: `test_pipeline.py`
