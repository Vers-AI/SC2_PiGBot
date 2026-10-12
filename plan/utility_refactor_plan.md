---
project: PiG_Bot
version: 0.1 (draft)
author: Praxis (from Drekken's bare plans, 2026-10-10)
goal: "Refactor three managers (Macro, Reaction, Scouting) to utility-AI decisions, coordinated through the intel system."
status: draft-for-review
priority: high
execution: All code changes via OpenCode (GLM 5.3) — Praxis plans/directs/verifies, does not code
tags:
  - StarCraft II
  - PiG_Bot
  - utility-AI
  - refactor
  - plan
---

# Utility AI Refactor Plan — PiG_Bot

**Source:** Drekken's directive, 2026-10-10. **Companion canvases:** `Pig_Bot_Archietecture.canvas` (system racetrack) and `Pig_Bot_Utility_Decision_Layer.canvas` (decision-layer view). **Theory:** Dave Mark & Kevin Dill, *Improving AI Decision Modeling Through Utility Theory* (GDC 2010) — slides + transcript in Readwise (tag: probability).

## Goal

Turn PiG_Bot's threshold-based decisions into scored decisions (utility AI) across three managers, with the **intel system as the coordinator**: one belief picture feeds every score, all managers read from the same intelligence.

**In scope:** Macro manager, Reaction manager, Scouting manager, intel wiring, counter-table elevation.
**Out of scope:** Opening build orders (scripted, stay as-is — MindMe's two-stage pattern), combat micro (already utility-shaped target scoring), the belief models themselves (they exist; they become inputs).

## Principles (from the GDC talk)

1. **Scores, not thresholds.** Every frame, all candidate actions get a number; the highest wins. No if-then gates, no sticky switches.
2. **Normalized scores make dissimilar things comparable** (Mark's "apples vs ammo"). Mining value, threat response, tech value — one currency.
3. **Response curves shape inputs.** Binary vs linear vs exponential vs logistic curves each produce different reaction personalities. Belief probabilities enter decisions through curves, not raw thresholds.
4. **Marginal utility runs execution.** The 1st worker pulled costs little; the 6th costs a lot. "How many" and "when to stop" are curve crossings, not constants.
5. **The architecture is hand-built; the magnitudes get fitted from telemetry** (2,300+ game corpus), not guessed. Baselines from domain knowledge (Liquipedia-annotated counter table), adjustments from our own games.

## Current state (the evidence)

- **Reaction manager:** fired `used_cheese_response` in 2/104 games (v0.13.1), 0/7 (v0.13.2). Baseline series vs lab villains: **0/6 responses fired across both before-tapes** — including games where belief hit **p_cheese = 0.983** (98% certain, zero action). Worker rush: bot believed "macro" at 79% while being killed (3/3 losses). Cannon rush: detection present, decision absent (3/3 wins on raw macro vs a weak constructed rush — borrowed luck).
- **Macro manager:** profile compositions + sticky one-way switches (economy switch never reverts) + threshold archon switch + counter/strategy/resource nudges patched on top. The counter table advises; it doesn't decide.
- **Scouting manager:** destination ranking is already VOI-based (staleness × impact) — half-utility already. Launch and production decisions remain rules.
- **Counter table (combat/target_scoring.py, 61 pairs):** Liquipedia-annotated quality, but **7 attackers / 33 targets only**. Missing: 11 Protoss units as attackers (entire air tree: Phoenix, Voidray, Oracle, Tempest, Carrier; plus DT, Mothership, Observer, Warp Prism — and Adept/Sentry, which the bot builds today). Mirror gaps: Dark Templar, Disruptor, Mothership uncovered as enemy targets.

## Phase 0 — Pre-work (blocking)

| Task | Owner | Notes |
|------|-------|-------|
| Extend COUNTER_TABLE to full Protoss arsenal (~80-100 new pairs, Liquipedia-annotated, same format) | OpenCode | The scoring function can only compare compositions the table knows. Blocking for Phase 1. |
| Rename `target_scoring` module → `counter_table` (elevation: combat-nudge → scoring core) | OpenCode | On Drekken's list; behavior-preserving rename + relocation of the table's role. |
| AGENTS.md setup: global (~/.config/opencode/AGENTS.md) + project (repo root) | Drekken provides universal; project adapted | Conventions, repo structure, what not to touch, verification workflow. |
| Verification harness prep: branch-run smoke (runner already supports `branch` param) | Praxis | Confirm branch-based runs work end-to-end before OpenCode's first PR. |

## Phase 1 — Macro manager (the flagship)

**Current:** `select_army_composition` — profile base → sticky econ switch → archon % threshold → nudges (counter-table, strategy, resource-pressure, priority reorder), clamped + re-normalized.

**Target:** composition candidates **compete by score** every macro cycle:

```
u(composition) = Σ counter_effectiveness(my_units vs believed_enemy_comp) × belief_weight
               + economy_term (worker count, income, saturation)
               + tech_term (progress toward upgrades/tech)
               − cost_term (minerals/gas/time to reach composition)
```

**Support units are not table rows — they're count curves (GDC principle #4).**
Units whose value is ability-driven (Sentry GS/FF today; later Raven/Medivac/
Queen-class) saturate with count. Flat per-enemy table bonuses mis-scale them
(the 5th Sentry scores like the 2nd). Their count-optimum emerges as a curve
crossing, computed per situation — never a hardcoded constant:

```
u_sentry(N) = w_ff × ff_choke_hold(N) × choke_threat            # caps at 1.0
            + w_gs × gs_uptime(N, T_fight) × army_size × enemy_ranged_frac
            − N × sentry_cost                                   # linear in stalkers
```

- Saturation terms cap at 1.0: `ff_choke_hold = min(1, N × regen × FF_DURATION / FF_COST)`;
  GS uptime similar over expected fight length `T_fight` (rolling average measured
  from our own engagements).
- The "right number" is where marginal value crosses marginal cost — situational by
  construction (ling flood at a ramp → ~6-8; open-field skirmish → ~2-3).
- Ability constants (FF_COST, GS_DURATION=11s, radius 4.5, energy regen) come from
  game data, not typed literals. Weights `w_ff`/`w_gs` are **fitted from telemetry**
  (principle #5): per-engagement logging of {sentry count, energy spent, GS uptime ×
  units covered, FF casts, trade ratio}, correlated against outcomes.
- `SITUATIONAL_ROWS` in target_scoring.py (commit 77dc34e) is the registry of these
  count-curve types: nudging must not scale them flat; Phase 1's scorer calls
  `marginal_utility(type, count, situation)` for registry types instead.
- Combat targeting keeps flat rows for registry types (a Sentry still prefers
  shooting what's in front of it) — the registry only governs production utility.

**Registry classification (research 2026-10-12) — entry criterion is "marginal
value per count is non-constant," not "has a fixed number":**

| Type | Curve shape | Live today | Nudge action now |
|---|---|---|---|
| Sentry | Fixed saturation (FF choke-hold, GS uptime) — independent of enemy count | 4 comps, up to 0.32 | Excluded from nudge (77dc34e) |
| High Templar | Saturation ∝ enemy clump size; Archon-merge floor on marginal cost; Feedback term vs casters | 4 comps, up to 0.4 | Keep in flat nudge until curve exists |
| Disruptor | Saturation ∝ clump size × micro capacity (controllable novas, cooldown-limited) | 5 comps, up to 0.3 | Keep in flat nudge until curve exists |
| Oracle | Harass-domain curve (worker-kill value × enemy econ state); saturates ~1-2 | Not built; no attacker rows | Design curve when first build adds it |
| Mothership | Binary N∈{0,1} (unique unit): cloaking/recall value vs cost | Not built; no rows | Binary build decision when first build adds it |

Why HT/Disruptor stay in the flat nudge while Sentry was excluded: their saturation
scales WITH enemy clump size, so the per-enemy table sum is a crude but
directionally-correct linearization (build AoE vs clumped bio). Sentry saturation
is enemy-count-independent, so flat scaling pushed it in a wrong direction. HT also
has a built-in overproduction valve — spent HTs merge into Archons, giving marginal
HTs residual value that marginal Sentries lack. Phase 1 replaces both flat
treatments with their curves.

- Candidates: the current profile compositions + counter-driven variants + a "hold/defend" candidate. The winner builds.
- The counter table becomes the core term of the score (no longer a post-decision nudge).
- Sticky switches die; composition flexes continuously with belief updates.
- **Fitted magnitudes:** counter-table values seeded from Liquipedia baseline, then fit against telemetry outcomes ("when we had immortals vs their roaches, what happened to win rate?").

## Phase 2 — Reaction manager

**Current:** category routing (CHEESE/ALL_IN/TIMING/MACRO) behind confidence thresholds; handlers fire or don't.

**Target:** every step, response candidates are **scored against continuing the plan**:

```
u(respond) = response_curve(p_cheese) × threat_severity(army_ratio, distance, time_to_contact)
u(continue) = plan_value(economy, tech progress)
```

- Response fires when `u(respond) > u(continue)` — no threshold to mis-set.
- **Marginal worker pulls:** each pulled worker's cost curve vs defensive value; pull count = the crossing point. "Pull enough, not everything."
- Response curves on belief inputs (logistic default; personality-tunable).
- Handlers (defend_worker_rush, defend_cannon_rush, ...) become candidate actions with costs, not privileged routines.
- **Success metric:** `used_cheese_response` telemetry flips from ~0% to measurable response rates, with survival-rate deltas in the lab.

## Phase 3 — Scouting manager

**Current:** VOI destination ranking (already scored); launch timing and scout production are rules.

**Target:** complete the utility:

```
u(scout mission) = expected_info_gain(VOI: staleness × decision_impact) − risk(scout loss) − cost(worker-time)
```

- Scout launches compete against mining/defending (same currency as everything else).
- Scouting feeds intel quality — which gates every other manager's score confidence. Scouting is the pump; intel is the reservoir.

## Phase 4 — Intel as the coordinator

The wiring phase: all three managers read the **same belief state** through the intel layer.

- **Intel system = single source of truth.** Strategy belief (P(cheese/all-in/timing/macro)), composition belief (fog-of-war decay), opponent memory (Dirichlet), intel quality (staleness) — one picture.
- Managers register as intel consumers; scores take belief-weighted inputs with quality gates (stale intel discounts its contribution, not zeroes it — decay, not binary).
- The reaction manager's score and the macro manager's score now share inputs: the same p_cheese that raises defense utility shifts composition candidates. **One belief picture, one coherent response** — the bot that pulls workers AND shifts production AND re-scouts, together, because all three read the same rising number.

## Verification (the test lab as harness)

- **Branch runs:** OpenCode works on `utility-refactor` branch; lab runs mount the branch worktree (runner supports `branch` param — verify in Phase 0).
- **Unit-level:** pure-Python tests pinning the utility curves before refactor touches them (OpenCode writes these first).
- **Behavior-level:** the banked before-tapes re-run on the branch:
  - Worker rush: PiG_Bot vs Who (WorkerRush) — baseline 3/3 losses, 0/3 responses. Target: responses fire, survival measurable.
  - Cannon rush: villain TBD (Chance/ProxyCannoneer recommended, consent-clean, pending Drekken's go) — vs the constructed Who CannonRush the detection reached 0.983 with zero action; target: action.
- **Regression:** full roster series (MicroMachine, Clicadinha, Who-random) master vs branch, compared in the lab's Compare tab.
- **Telemetry gates:** `used_cheese_response` firing rate, response latency, win-rate deltas per matchup. No regression vs master without a documented reason.

## Division of labor

| Role | Owner |
|------|-------|
| Plan, specs, direction, verification | Praxis |
| All code changes | OpenCode (GLM 5.3), per Drekken's directive |
| AGENTS.md (universal + project), milestone review, design calls | Drekken |
| Lab runs, telemetry analysis, Compare reports | Praxis |

**Praxis does not touch repo code directly.** Spec updates and failing-test reports are the course-correction mechanism.

## Evidence appendix (before-tape, banked 2026-10-10)

| Series | Result | Responses fired | Belief peak | Story |
|--------|--------|-----------------|-------------|-------|
| vs Who WorkerRush (979-981) | 3/3 losses | 0/3 | p_macro 0.795 (confidently wrong) | "79% sure it was macro while being murdered by workers" |
| vs constructed cannon rush (986-988) | 3/3 wins | 0/3 | **p_cheese 0.983** (certain, inert) | "98% sure, did nothing, won anyway" |

Both series: LeyLinesAIE_v3, full replays banked in the lab. Post-refactor re-runs = the after-tape.