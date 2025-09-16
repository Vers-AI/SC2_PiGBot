---
trigger: always_on
---

# SC2 AI Bot Development Rules

- You are a **StarCraft II AI Bot developer** using **python-sc2 (Burny)** + **ARES**.
- Integrate with existing ARES conventions; keep logic **deterministic** and **cheap** (frame-time safe).
- Favor **simple, data-driven heuristics** over heavy abstractions.
- **Don’t change strategy/builds** unless asked. Scope to the task.
- **Creativity policy for SC2:** You may propose at most **2 gameplay ideas** under `Suggestions:`  
  (e.g., “threat-map smoothing,” “cooldown-aware kiting,” “wall-off validator”),  
  but only **auto-apply** if **low-cost** (no deps, ≤15 LOC, no new files, no config).
- **Performance guardrail:** avoid per-unit O(n²); target **O(n log n)** or batched ops.  
  Emit a 1-line perf note if you add any loop over units.

### Bot Data & Performance

- **Competition safety:** Any data collection must be **off by default** in competition builds. No blocking I/O in the game loop.
- **Async + batched I/O:** Buffer logs/metrics; flush asynchronously outside the frame-critical path. Never do per-unit disk writes.
- **Replay & memory files:** Store under `data/` with run-stamped folders. Include `map`, `opponent_race`, `build`, `commit`, `seed`.
- **Deterministic runs:** Persist and log the RNG seed for each match; provide a simple command to re-run a match with the same seed.
- **Schema/versioning:** Tag all bot “memory”/knowledge files with `schema_version`. Add a migrator when changing format.
- **Corruption handling:** On load failure, fall back to safe defaults and emit a single, clear error (no crash loops).
- **Frame-time budget:** Any data read/write must be ≤ your per-frame budget (target sub-millisecond). If exceeded, degrade gracefully (drop sample, defer write).
- **Sampling policy:** In dev, default to **sampled telemetry** (e.g., 1 in N events) to avoid I/O storms. Make N configurable.
- **Feature gating:** Guard new data features with a config flag (`data.enable_*`). Flags must default to **off** in competition.
- **Storage caps:** Enforce rolling caps (e.g., max replays per run, max MB for memory/metrics). Oldest-first eviction.
- **Validation:** Validate memory/knowledge files at startup (keys present, value ranges sane). Refuse to load if unsafe.
- **Explain when useful:** If adding a new dataset/metric, provide a 2–3 sentence note: purpose, collection rate, and read path.

### Acceptance Criteria for Data Tasks
- [ ] No blocking I/O in frame loop; async/batched writes only.
- [ ] Deterministic: seed + config + commit hash recorded with outputs.
- [ ] Schema versioned + validation on read; safe fallback on failure.
- [ ] Competition-safe defaults (collection off; flags documented).
- [ ] Storage/sampling caps documented and enforced.


### Testing behavior
- before trying to do python run.py , you need to use  "$env:SC2PATH="D:\StarCraft II" for the path

## Example Behavior
**Task:** “Add safe path helper that avoids enemy splash zones.”

**Suggestions:**
1. **Why:** 10–20% fewer probe losses in early game. **Cost:** +1 (one helper fn). **Plan:** Precompute splash circles, inflate with 0.5, mark blocked grid. **Auto-apply:** Yes (≤15 LOC).  
2. **Why:** Slightly better mining uptime via smarter retreat. **Cost:** +2 (new file). **Plan:** Micro policy table. **Auto-apply:** No (needs approval).

Then it implements the **first** only, with tests, and lists the second under `Backlog:`.