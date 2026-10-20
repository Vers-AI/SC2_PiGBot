---
project: PiG_Bot
version: 0.3 (trimmed for LLM use)
author: Praxis
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

## Goal

Turn PiG_Bot's threshold-based decisions into scored decisions (utility AI) across three managers, with the **intel system as the coordinator**: one belief picture feeds every score, all managers read from the same intelligence.

**In scope:** Macro manager, Reaction manager, Scouting manager, intel wiring, counter-table elevation.
**Out of scope:** Opening build orders from the build runner, combat micro (already utility-shaped target scoring), the belief models themselves (they become inputs).

## Principles

1. **Scores, not thresholds.** Every decision cycle, all candidate actions get a number; the highest wins. No if-then gates, no sticky switches.
2. **Normalized scores make dissimilar things comparable.** Mining value, threat response, tech value — one currency.
3. **Response curves shape inputs.** Binary vs linear vs exponential vs logistic curves each produce different reaction personalities. Belief probabilities enter decisions through curves, not raw thresholds.
4. **Marginal utility runs execution.** The 1st worker pulled costs little; the 6th costs a lot. "How many" and "when to stop" are curve crossings, not constants.
5. **The architecture is hand-built; the magnitudes get fitted from telemetry** (2,300+ game corpus), not guessed. Baselines from domain knowledge (Liquipedia-annotated counter table), adjustments from our own games.
6. **Curve steepness stabilizes.** Exponential and logistic curves convert raw ratios into ordinal priority: distant noise gets ignored, commitment gets aggressive near critical values. This is what replaces sticky switches. Scored hysteresis is the fallback if thrash survives the curves.
7. **Scores are expected utilities.** Beliefs multiply, they don't add: value × confidence. A choice is only as valuable as the belief behind it, so stale intel discounts a score instead of zeroing it.
8. **Perceive and decide stay separated.** The intel layer perceives; the managers decide. Scoring reads the intel snapshot only, never raw game state.

## Design language (how scores are built)

- **Reasoner** = a manager's decision step. **Choices** = its candidates (compositions, responses, scout missions).
- **Consideration** = one scored aspect of the situation (economy, threat, cost, staleness). Small, parameterized, reusable across managers. Two functions: `load(params)` and `evaluate(context) → appraisal`.
- **Context** = the intel snapshot handed to considerations. Perception belongs to the intel layer; considerations never dig into raw game state.
- **Appraisal** = (base score, multiplier, veto).

Canonical combination, used by every task:

```
score(choice) = [ Σ base_scores ] × [ Π multipliers ]   (any veto fires ⇒ score 0)
```

- Every raw input passes a response curve first: normalize to [0,1], then shape (linear / exponential / logistic). No raw values enter scores.
- Beliefs enter as multipliers, not additive terms. Scores are expected utilities: value × confidence.
- Combine at most 2-3 considerations per box, then chain boxes (compartmentalized confidence). Reuse boxes across managers: threat severity feeds reaction scoring AND scouting risk.
- Hard impossibilities veto (no production facility for a candidate composition, no worker available to scout). Costs are negative bases, never vetoes.

Curve policy:

- Exponential / logistic curves are the default for threat-like inputs: distant noise ignored, aggressive commitment near critical. **Curve steepness is the primary anti-oscillation tool.**
- If tests still show thrash, add a commitment consideration (hysteresis): the current winner takes a scored bonus or margin. Scored, never sticky.
- Curve parameters live in config. They are personality hypotheses, fitted from data in Task 7, not hand-guessed forever.

Update cadence:

- Re-score on belief change or every N game loops (N in config). Cache appraisals until an input changes. Never score every frame.

Selection rule: deterministic argmax for this refactor (verification needs determinism). Weighted-random top-N is a future personality knob — do not add it now.

## Global constraints (apply to every task)

Style, dependency discipline, and scope rules live in AGENTS.md — not repeated here. These are the refactor-specific constraints:

1. Do not touch: opening build orders, combat micro logic, belief models — they are inputs, not targets. The counter table module is in scope (Tasks 1-2); the combat micro that consumes it is not.
2. Every scoring change requires pure-Python unit tests pinning the scoring math BEFORE wiring it into the manager (test-first).
3. Every behavior change ships with its telemetry gate result; no regression vs master without a documented reason.
4. Preserve existing telemetry event names and fields. Add new ones freely; remove or rename nothing.
5. One task, one PR. PR description must state which acceptance criteria are met and which remain.
6. Every score follows the Design language: intel context only (no raw game-state access), every input through a response curve, re-score on cadence — never per frame.

## Task 1 — Rename + elevate counter table

**Goal:** `target_scoring` module renamed to `counter_table`; its role documented as scoring core, not combat nudge.

**Scope:** module rename, import updates, docstring stating the table's elevated role.
**Preserve:** behavior — zero functional change.
**Acceptance:**
- [ ] old module path gone, `counter_table` imports resolve
- [ ] unit tests pass unchanged
- [ ] full roster smoke run matches master within noise

## Task 2 — Extend COUNTER_TABLE to full Protoss arsenal

**Current:** 61 pairs in `combat/target_scoring.py`, Liquipedia-annotated, but only 7 attackers / 33 targets.

**Goal:** fill coverage gaps so any composition the bot builds or faces scores correctly. The scoring function can only compare compositions the table knows — blocking for Task 3.

**Gaps to cover (from audit):**
- 11 missing Protoss attackers: Phoenix, Voidray, Oracle, Tempest, Carrier, DT, Mothership, Observer, Warp Prism, Adept, Sentry
- Missing enemy targets: Dark Templar, Disruptor, Mothership

**Rules:** ~80-100 new pairs, Liquipedia-annotated, same format as existing entries.
**Acceptance:**
- [ ] every unit the bot can build has complete target rows
- [ ] table passes format-validation test (add one: no duplicate pairs, all values in same scale)
- [ ] no existing pair values modified

## Task 3 — Macro manager utility scoring

**Current:** `select_army_composition` — profile base → sticky econ switch → archon % threshold → nudges (counter-table, strategy, resource-pressure, priority reorder), clamped + re-normalized.

**Goal:** composition candidates compete by score every macro cycle.

**Design (canonical form — see Design language):**
```
score(comp) = [ Σ counter_effectiveness(my_units vs believed_enemy_comp)
              + economy_term(worker count, income, saturation)
              + tech_term(upgrade/tech progress)
              − cost_term(minerals/gas/time) ] × belief_confidence
```
- Counter table becomes the core base score (no longer a post-decision nudge).
- `belief_confidence` is a multiplier: expected utility, not an additive term.
- Candidates: current profile compositions + counter-driven variants + hold/defend candidate, capped at 8 per cycle. The winner builds.
- Remove sticky one-way switches and the threshold archon switch. Stabilizers: curve steepness first, commitment consideration second — never hard switches.
- Magnitudes: seed from Liquipedia baselines, fit later from telemetry (Task 7); document seed provenance in comments.

**Acceptance:**
- [ ] unit tests pin curve shapes and candidate scoring before integration
- [ ] econ switch removed, no one-way state remains
- [ ] branch passes full roster regression vs master in Compare tab

## Task 4 — Reaction manager utility scoring

**Current:** category routing (CHEESE/ALL_IN/TIMING/MACRO) behind confidence thresholds; handlers fire or don't. Baseline: `used_cheese_response` fired 2/104 games (v0.13.1), 0/7 (v0.13.2), 0/6 in banked tapes — including one game at p_cheese 0.983 with zero action.

**Goal:** replace category routing + confidence thresholds with scored response vs continue.

**Design:**
```
u(respond) = response_curve(p_cheese) × threat_severity(army_ratio, distance, time_to_contact)
u(continue) = plan_value(economy, tech progress)
```
- Response fires when `u(respond) > u(continue)` — no threshold to mis-set.
- Worker pulls: marginal cost curve vs defensive value; pull count = curve crossing point. Pull enough, not everything.
- Logistic response curve as default, personality-tunable parameter exposed.
- Handlers (`defend_worker_rush`, `defend_cannon_rush`, ...) become scored candidates, not privileged routines.
- Commitment: `u(continue)` may carry a hysteresis bonus so near-ties don't thrash between respond and continue. Scored, never sticky.

**Success metric:** `used_cheese_response` firing rate ≥ measurable on rush tapes (baseline 0/6), with survival-rate delta reported.

## Task 5 — Scouting manager utility scoring

**Current:** VOI destination ranking is already scored; launch timing and scout production are rules.

**Goal:** complete utility on launch and production decisions.

**Design:**
```
u(scout) = info_gain(staleness × decision_impact) × P_survive − cost(worker-time)
```
- Risk enters as a survival multiplier (expected value of the info), cost as a negative base — canonical form.
- Scout launches compete against mining/defending in the same score currency.

**Acceptance:**
- [ ] existing VOI destination ranking unchanged
- [ ] launch/production decisions routed through the same scoring currency
- [ ] unit tests pin the risk and info-gain terms

## Task 6 — Intel coordination wiring

**Goal:** single belief state read by all three managers through the intel layer.

**Design:**
- Managers register as intel consumers.
- Stale intel discounts contribution (decay), never zeroes it.
- Same `p_cheese` input feeds reaction and macro scores.
- A rising p_cheese must raise defense utility, shift composition, and raise scout priority together — all three read the same number.

**Acceptance:**
- [ ] no manager reads a private/duplicate belief source
- [ ] staleness gate tested (decay curve, not binary cutoff)
- [ ] full regression vs master

## Task 7 — Fitted magnitudes from telemetry

**Goal:** reseed score weights using lab outcomes instead of Liquipedia seeds alone.

**Method:** fit per-term weights against the 2,300+ game corpus (e.g., immortal-vs-roach outcomes vs win rate).
**Acceptance:**
- [ ] fitted weights documented with fit evidence
- [ ] regression run shows ≥ no-worse across roster; improvements logged per matchup

## Verification (the test lab as harness)

- **Branch runs:** OpenCode works on `development` branch; lab runs mount the branch worktree (runner `branch` param — confirmed in pre-work).
- **Unit-level:** pure-Python tests pinning the utility curves before refactor touches them (OpenCode writes these first).
- **Behavior-level:** the banked before-tapes re-run on the branch:
  - Worker rush: PiG_Bot vs Who (WorkerRush) — baseline 3/3 losses, 0/3 responses. Target: responses fire, survival measurable.
  - Cannon rush: villain TBD (Chance/ProxyCannoneer proposed) — vs the constructed Who CannonRush the detection reached 0.983 with zero action; target: action taken.
- **Regression:** full roster series (MicroMachine, Clicadinha, Who-random) master vs branch, compared in the lab's Compare tab.
- **Telemetry gates:** `used_cheese_response` firing rate, response latency, win-rate deltas per matchup. No regression vs master without a documented reason.

## Baselines (banked 2026-10-10)

| Series | Result | Responses fired | Belief peak |
|--------|--------|-----------------|-------------|
| vs Who WorkerRush (979-981) | 3/3 losses | 0/3 | p_macro 0.795 |
| vs constructed cannon rush (986-988) | 3/3 wins | 0/3 | p_cheese 0.983 |
