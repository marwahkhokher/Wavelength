# Parallel Execution & Interface Contract
**Project**: Confidence Building Platform (Voice Pipeline)  
**Version**: 2.1 (SHARED LLM ARCHITECTURE & SEQUENTIAL PIPELINE)  
**Status**: Approved for Team Parallel Development  

---

## 1. Flow Architecture & Component Pipeline

The architecture follows your exact sequential pipeline: **Qwen performs in-depth answer evaluation per turn first, and passes its evaluation output to the Prompt Generation LLM** so the conversation AI can respond with complete awareness of the user's answer quality, structure, and tone.

```
                                [ User Speech Utterance ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
Areej: Whisper Small STT                                       Zaid: emotion2vec / wav2vec2
+ Timestamped Filler Word Detection                             + Audio Tone & Emotion Analysis
+ Python Word Count & Utterance Duration                        + Speech Rate (WPM) & Pause Calculation
        │                                                                       │
        │ STTResult                                                             │ ToneResult
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │ Ahmed: Qwen Deep Eval │
                                ├───────────────────────┤
                                │ Deep Answer Analysis: │
                                │ Clarity, Confidence,  │
                                │ Structure, Relevance, │
                                │ Fluency & Filler score│
                                └───────────┬───────────┘
                                            │
                                            │ QwenEvaluationResult
                                            ▼
                                ┌───────────────────────┐
                                │ Taha: Prompt Gen LLM  │
                                ├───────────────────────┤
                                │ Consumes:             │
                                │ 1. Qwen Evaluation    │
                                │ 2. STT Transcript     │
                                │ 3. Tone & Emotion     │
                                │ 4. Persona & Scenario │
                                │ 5. History            │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │ Taha: ElevenLabs TTS  │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                   Audio Output to User
                                            │
                                            ▼ (Session End)
                                ┌───────────────────────┐
                                │   End of Session      │
                                ├───────────────────────┤
                                │ 1. Evaluation Screen  │ ◄── Qwen Metrics (Clarity: 8/10, etc.)
                                │ 2. Coaching Page      │ ◄── LLM Answer Refinement & Actionable Tips
                                └───────────────────────┘
```

---

## 2. Detailed Interface Specifications by Owner

### 👤 Armeen: Scenario + Persona + Session Context & Baseline Store
* **Output Artifact**: `SessionContext.json`
* **Responsibilities**:
  1. Store initial baseline metrics from the onboarding questionnaire.
  2. Parse scenario & persona setup (mode: `"professional"` vs `"personal"`).
  3. Package persona personality, difficulty, duration, and baseline metrics into `SessionContext`.

#### Schema (`SessionContext.json`):
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
  },
  "baseline_metrics": {
    "clarity": 6.5,
    "confidence": 6.0,
    "structure": 5.5,
    "relevance": 7.0,
    "fluency": 6.0,
    "filler_words_score": 5.0
  }
}
```

---

### 👤 Areej: Whisper Small + Filler Detection
* **Output Artifact**: `STTResult`
* **Responsibilities**: Transcribe audio, extract timestamped filler words (`"um"`, `"uh"`, `"like"`), compute word counts and utterance duration.

#### Schema (`STTResult`):
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
  "utterance_duration_sec": 3.8,
  "stt_latency_ms": 210.0
}
```

---

### 👤 Zaid: emotion2vec / wav2vec2 Tone & Acoustic Analysis
* **Output Artifact**: `ToneResult.json`
* **Responsibilities**: Extract tone emotion state, pitch/energy variation, hesitation score, speech pauses, and calculate Words Per Minute (WPM) via Python script.

#### Schema (`ToneResult.json`):
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

---

### 👤 Ahmed: Qwen Deep Evaluation + Final Evaluation + Coaching Page
* **Output Artifacts**:
  1. `QwenEvaluationResult.json` (Per-turn deep evaluation output passed to Prompt LLM)
  2. `FinalSessionEvaluation.json` (End-of-session 1–10 metrics scorecard)
  3. `CoachingPageReport.json` (Answer refinement & actionable improvement plan)
* **Responsibilities**:
  1. **Per-Turn Deep Evaluation (Passed to Prompt LLM)**: Qwen analyzes the turn deeply based on confidence, clarity, structure, relevance, fluency, filler-word usage, WPM, and pacing.
  2. **Final Session Scorecard**: Generates 1–10 metric scores (`"Clarity: 8/10"`, `"Confidence: 7/10"`, etc.) comparing final scores against onboarding baseline metrics.
  3. **Coaching Page Engine**: Analyzes complete session, refines user answers ("Before vs After" rewrites), and explains what & how to improve.

#### Schema (`QwenEvaluationResult.json` - Passed to Prompt LLM per turn):
```json
{
  "turn_index": 2,
  "deep_metrics": {
    "clarity": 7.8,
    "confidence": 6.0,
    "structure": 6.5,
    "relevance": 9.0,
    "fluency": 7.0,
    "filler_words_score": 6.5,
    "overall_turn_score": 7.1
  },
  "answer_analysis": {
    "main_point_detected": "User claimed pipeline refactoring increased team velocity.",
    "structural_assessment": "Weak initial structure due to filler words ('um', 'like').",
    "emotional_alignment": "Speech was hesitant; tone showed uncertainty.",
    "key_flaw": "Vague quantitative claims without supporting metrics."
  },
  "suggested_conversation_followup_direction": "Challenge the user on specific velocity numbers or metrics."
}
```

#### Schema (`FinalSessionEvaluation.json`):
```json
{
  "session_id": "sess_987654",
  "total_turns": 5,
  "metrics_display_10_scale": {
    "clarity": "8/10",
    "confidence": "7/10",
    "structure": "6/10",
    "relevance": "9/10",
    "fluency": "7/10",
    "filler_words_score": "8/10"
  },
  "overall_session_score": "7.5/10",
  "improvement_vs_baseline": {
    "clarity_delta": "+1.5",
    "confidence_delta": "+1.0",
    "structure_delta": "+0.5",
    "relevance_delta": "+2.0",
    "fluency_delta": "+1.0",
    "filler_words_delta": "+3.0"
  }
}
```

#### Schema (`CoachingPageReport.json`):
```json
{
  "session_id": "sess_987654",
  "summary_feedback": "Strong domain expertise and relevance, but answer structure dropped under tough questioning.",
  "answer_refinements": [
    {
      "turn_index": 2,
      "original_user_answer": "Um, I believe my contribution to the project, like, increased team velocity significantly.",
      "refined_better_answer": "Over the last quarter, my work on pipeline refactoring directly increased team velocity by 25%.",
      "why_it_is_better": "Eliminates filler words ('um', 'like') and replaces vague assertions with concrete quantitative metrics.",
      "key_skill_boosted": "Clarity & Confidence"
    }
  ],
  "improvement_action_plan": [
    {
      "area": "Structure",
      "current_level": "6/10",
      "target_level": "8/10",
      "issue_identified": "Hesitated and lost structure when manager pushed back on compensation.",
      "actionable_how_to": "Use the STAR method (Situation, Task, Action, Result) to structure responses immediately before answering."
    }
  ]
}
```

---

### 👤 Taha: Prompt-Gen LLM + ElevenLabs + Pipeline Orchestrator
* **Output Artifact**: `PromptLLMOutput` & Streaming ElevenLabs Audio.
* **Responsibilities**:
  1. Assembles input payload for Prompt LLM containing:
     - `session_context`: Persona profile, scenario context, mode rules (Armeen)
     - `current_stt`: Transcript + filler word count (Areej)
     - `current_tone`: Primary emotion, WPM, pauses (Zaid)
     - `qwen_evaluation`: Deep answer analysis & metric scores (Ahmed)
     - `conversation_history`: Previous turns
  2. Prompt Generation LLM uses **Qwen's deep answer analysis** to generate the next conversation question/response in character.
  3. Stream text output to ElevenLabs TTS for real-time audio playback.
  4. VAD Interruption: Instantly halt ElevenLabs playback if user interrupts mid-sentence ("Cut AI off").

---

## 3. Shared LLM Integration Strategy (Roleplay vs Coaching Modes)

To simplify system complexity and API integrations, the **same underlying LLM provider/family** (e.g. Qwen / GPT-4o / Claude) is utilized across two operational modes:

1. **Conversation Roleplay Mode (Taha - Live Session)**:
   * **System Prompt**: *"You are playing [Persona Name]. Respond to the candidate's utterance using Qwen's deep evaluation report, maintaining persona tone (Professional vs Personal)."*
2. **Coaching & Refinement Mode (Ahmed - Post Session)**:
   * **System Prompt**: *"You are an expert executive communications coach. Analyze the full session transcript and evaluations to generate 'Before vs After' answer refinements and actionable improvement recommendations."*
   * **Zero Live Latency Penalty**: Runs post-session offline when the user opens the Coaching Page.

---

## 4. Sequential Verification Matrix

| Step | Component / Owner | Inputs Consumed | Output Produced | Next Dependent Step |
| :--- | :--- | :--- | :--- | :--- |
| **1. STT & Fillers** | Areej | User Audio | `STTResult` (Transcript, fillers) | Step 3 (Qwen Eval) |
| **2. Tone & Acoustic** | Zaid | User Audio | `ToneResult` (Emotion, WPM, pauses) | Step 3 (Qwen Eval) |
| **3. Qwen Deep Eval** | Ahmed | `STTResult` + `ToneResult` + Python Stats | `QwenEvaluationResult` (Deep analysis) | Step 4 (Prompt LLM) |
| **4. Prompt LLM** | Taha | `QwenEvaluationResult` + `SessionContext` + History | Persona Response Text | Step 5 (ElevenLabs TTS) |
| **5. Voice Output** | Taha | Persona Response Text | Audio Stream to User | Next Turn or Session End |
| **6. End Evaluation** | Ahmed | All `QwenEvaluationResult` items | 1–10 Metric Scorecard & Coaching Page | Displayed to User |
