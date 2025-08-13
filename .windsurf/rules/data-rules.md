---
trigger: model_decision
description: when applying data to the bot
---

# Data Rules (Conditional: when working with data/telemetry)

## General Data & Telemetry
- **Determinism first:** Capture RNG seed, code version, and config with any saved artifact so runs are reproducible.
- **Schema discipline:** Version all data schemas (`schema_version` field). Backward-compatible changes only; otherwise add a migrator.
- **Validation:** Validate inputs before use (shape, types, required keys). Fail fast with actionable errors.
- **No secrets in data/logs:** Never log API keys, tokens, or sensitive paths. Redact by default.
- **Minimal logging:** Log events, not streams. Prefer structured logs (JSON lines). Default level = `INFO`; use `DEBUG` only in dev.
- **Env boundaries:** Keep dev/test/prod data isolated (separate dirs/buckets). Never read/write prod data in dev/test.
- **File formats:** Prefer portable, line-oriented formats (JSONL/CSV). Use binary only with explicit justification.
- **Size & retention:** Set per-artifact size caps and a retention window. Auto-prune old artifacts.
- **Explain when useful:** For non-trivial data changes, include 2–3 sentences on trade-offs and expected read/write volume.

## SC2 Bot-Specific Data Rules
- **Competition safety:** Any data collection must be **off by default** in competition builds. No blocking I/O in the game loop.
- **Async + batched I/O:** Buffer logs/metrics; flush asynchronously outside the frame-critical path. Never do per-unit disk writes.
- **Replay & memory files:** Store under `data/` with run-stamped folders. Include map, opponent_race, build, commit, seed.
- **Deterministic runs:** Persist and log the RNG seed for each match; provide a simple command to re-run a match with the same seed.
- **Schema/versioning:** Tag all bot “memory”/knowledge files with `schema_version`. Add a migrator when changing format.
- **Corruption handling:** On load failure, fall back to safe defaults and emit a single, clear error (no crash loops).
- **Frame-time budget:** Any data read/write must be ≤ your per-frame budget (target sub-millisecond). If exceeded, degrade gracefully (drop sample, defer write).
- **Sampling policy:** In dev, default to **sampled telemetry** (e.g., 1 in N events) to avoid I/O storms. Make N configurable.
- **Feature gating:** Guard new data features with a config flag (`data.enable_*`). Flags must default to **off** in competition.
- **Storage caps:** Enforce rolling caps (e.g., max replays per run, max MB for memory/metrics). Oldest-first eviction.
- **Validation:** Validate memory/knowledge files at startup (keys present, value ranges sane). Refuse to load if unsafe.
- **Explain when useful:** If adding a new dataset/metric, provide a 2–3 sentence note: purpose, collection rate, and read path.

## Acceptance Criteria for Data Tasks
- [ ] No blocking I/O in frame loop; async/batched writes only.
- [ ] Deterministic: seed + config + commit hash recorded with outputs.
- [ ] Schema versioned + validation on read; safe fallback on failure.
- [ ] Competition-safe defaults (collection off; flags documented).
- [ ] Storage/sampling caps documented and enforced.
