# Wavelength Backend

FastAPI backend for the Confidence Building Platform (see PRD v1.0). Requirement IDs
in code comments (`FR-AUTH-1`, `FR-SESS-4`, etc.) map directly back to the PRD so
you can trace any endpoint to its spec line.

## 1. Local setup

```bash
cd backend
cp .env.example .env
# fill in JWT_SECRET_KEY and OPENAI_API_KEY at minimum

docker compose up --build
```

This starts the API on `localhost:8000`, Postgres on `5432`, Redis on `6379`.
Check `http://localhost:8000/health` and `http://localhost:8000/docs` (Swagger UI,
auto-generated - useful for the frontend team to see every request/response shape).

## 2. Run the first migration

```bash
docker compose exec api alembic revision --autogenerate -m "initial schema"
docker compose exec api alembic upgrade head
```

From here on, any model change: `alembic revision --autogenerate -m "..."` then
`alembic upgrade head`.

## 3. Run tests

```bash
docker compose exec api pytest
```

## 4. Project layout

```
app/
  main.py            entrypoint, wires all routers
  config.py           env-driven settings
  database.py          SQLAlchemy engine/session
  models/              ORM models (one file per entity) + enums.py
  schemas/              Pydantic request/response shapes
  routers/              one file per screen area (auth, onboarding, home, sessions, admin)
  services/              persona generation, evaluation scoring, coach recs, tone rules
  core/                  security (JWT/hashing), auth dependencies
  ws/                    live session WebSocket gateway
alembic/                 migrations
tests/
```

## 5. Build order (what to do next, in sequence)

1. **Fill in `.env`** with a real `OPENAI_API_KEY` and a generated `JWT_SECRET_KEY`.
2. **Run the first migration** (step 2 above) and confirm tables exist:
   `docker compose exec db psql -U wavelength -d wavelength -c '\dt'`
3. **Smoke-test the auth + onboarding + home flow** end to end via `/docs`:
   signup -> get onboarding questions -> submit answers -> check `/home` shows
   baseline metrics.
4. **Smoke-test the session flow**: create session (mode) -> set scenario ->
   generate persona -> edit persona -> finalize -> set difficulty/duration ->
   begin. This is the part that most depends on ai-prompt-engineering's actual
   persona-generation prompt quality - `app/services/persona_generation.py` is
   the one function to iterate on with them.
5. **Confirm the voice provider with voice-tech-infra** before building past
   the WebSocket stub in `app/ws/live_session.py`. FR-SESS-4 (interrupting the
   AI mid-speech) is the requirement that decides this - it needs a realtime
   streaming provider with barge-in support, not a request/response API.
6. **Wire the chosen voice provider into `VoiceProvider`** in `live_session.py`,
   and start appending real transcript turns instead of the stubbed list.
7. **Deploy**: the GitHub Actions workflow at `.github/workflows/backend-ci.yml`
   runs tests on every push touching `backend/`. Add a deploy job once you've
   picked a host (Render/Fly/EC2/etc.) - not included here since that's an
   infra decision, not a code one.

## 6. Open items this backend deliberately left flexible

These map to PRD Section 9 ("Items Awaiting Confirmation"). Nothing here blocks
building - the schema is designed so each of these can be answered later
without a breaking migration:

- **Full metric list** (#1) - `app/models/enums.py::CONFIRMED_METRIC_KEYS`.
  Metrics are stored as JSON everywhere (questionnaire, snapshots, evaluations),
  so adding a metric is a one-line change to that list plus a prompt update in
  `services/evaluation.py`, not a schema migration.
- **Exact questionnaire + answer-to-metric mapping** (#2) -
  `app/services/questions.py`. Swap the `QUESTIONS` list once product-research
  finalizes it.
- **Persona profile fields** (#3) - `app/services/persona_generation.py::PERSONA_PROFILE_FIELDS`.
  Stored as JSON on the session, so the field list can grow without a migration.
- **Difficulty levels** (#4) - `ConversationSession.difficulty` is a free string,
  not an enum, on purpose - lock it down to an `Enum` once the levels are fixed.
- **Duration options / extension amount** (#5) - same reasoning, stored as
  plain integers (seconds).
- **"Current performance" definition** (#6) - `routers/home.py` currently
  returns the latest metric snapshot. Full history is kept in `metric_snapshots`,
  so switching to a running average is a query change, not a data change.
- **Language support** (#7) - not addressed yet; depends on the voice
  provider chosen for live sessions (item 5 above).
- **Admin dashboard fields / retention definition** (#8) - `routers/admin.py`
  implements the simplest reasonable retention definition (active-in-window /
  total). Flag to product before treating this number as real.
