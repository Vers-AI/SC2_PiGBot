---
trigger: model_decision
description: When working with data
---

### Data & Telemetry (Global)

- **Determinism first:** Capture `RNG seed`, `code version`, and `config` with any saved artifact so runs are reproducible.
- **Schema discipline:** Version all data schemas (`schema_version` field). Backward-compatible changes only; otherwise add a migrator.
- **Validation:** Validate inputs before use (shape, types, required keys). Fail fast with actionable errors.
- **No secrets in data/logs:** Never log API keys, tokens, or sensitive paths. Redact by default.
- **Minimal logging:** Log events, not streams. Prefer structured logs (JSON lines). Default level = `INFO`; use `DEBUG` only in dev.
- **Env boundaries:** Keep dev/test/prod data isolated (separate dirs/buckets). Never read/write prod data in dev/test.
- **File formats:** Prefer portable, line-oriented formats (JSONL/CSV). Use binary only with explicit justification.
- **Size & retention:** Set per-artifact size caps and a retention window. Auto-prune old artifacts.
- **Explain when useful:** For non-trivial data changes, include 2–3 sentences on trade-offs and expected read/write volume.