---
project: PiG_Bot
version: 0.5
author: Praxis
goal: "Lay the bot-wide utility-AI foundation (reasoner library + intel context), then convert Macro, Reaction, and Scouting as first adopters. Phases 2-3 extend to combat, structures, expansion, micro."
status: draft-for-review
priority: high

---

# Utility AI Refactor Plan — PiG_Bot

## Goal

Lay the utility-AI foundation for the whole bot — one shared reasoner over one intel picture — then convert the three managers as its first adopters. Every later decision system (combat, structures, expansion) joins the same currency instead of growing new thresholds.

**Phase 1 (this plan):** scoring-table elevation → reasoner library + intel context → Macro, Reaction, and Scouting conversions → coordination wiring → fitted magnitudes.
**Phase 2-3 (Adoption Roadmap at the end):** combat engagement, chrono/recharge, expansion, defender allocation, mass recall, macro leftovers, per-unit micro.
**Out of scope for Phase 1:** opening build orders from the build runner, combat micro execution — runtime unit targeting/maneuver behavior (mechanical import/comment updates in micro files are in scope), the belief models themselves (they become inputs).

## Design language

Every decision cycle, all candidate actions get a number; the highest wins. No if-then gates, no sticky switches.

- **Reasoner** = a manager's decision step. **Choices** = its candidates (compositions, responses, scout missions).
- **Consideration** = one scored aspect of the situation (economy, threat, cost, staleness). Small, parameterized, reusable across managers. Two functions: `load(params)` and `evaluate(context) → appraisal`.
- **Context** = the intel snapshot handed to considerations. Perception belongs to the intel layer; considerations never dig into raw game state.
- **Appraisal** = (base score, multiplier, veto).

Canonical combination, used by every task:

```
score(choice) = [ Σ base_scores ] × [ Π multipliers ]   (any veto fires ⇒ score 0)
```

- One currency: normalize raw inputs so mining value, threat response, and tech value stay comparable.
- Every raw input passes a response curve first: normalize to [0,1], then shape (linear / exponential / logistic). No raw values enter scores.
- Beliefs enter as multipliers, not additive terms — scores are expected utilities (value × confidence). Stale intel discounts a score instead of zeroing it.
- Marginal utility decides "how many": the 1st worker pulled costs little, the 6th costs a lot. Pull counts and stop points are curve crossings, not constants.
- Combine at most 2-3 considerations per box, then chain boxes (compartmentalized confidence). Reuse boxes across managers: threat severity feeds reaction scoring AND scouting risk.
- Hard impossibilities veto (no production facility for a candidate composition, no worker available to scout). Costs are negative bases, never vetoes.
- Exponential / logistic curves are the default for threat-like inputs: distant noise ignored, aggressive commitment near critical. **Curve steepness is the primary anti-oscillation tool.** Scored hysteresis is the fallback if thrash survives the curves — never sticky switches.
- Curve parameters live in config as personality hypotheses, seeded from the Liquipedia-annotated table and fitted from telemetry in Task 8 (2,300+ game corpus).

Update cadence:

- Re-score on belief change or every N game loops (N in config). Cache appraisals until an input changes. Never score every frame.

Selection rule: deterministic argmax for this refactor (verification needs determinism). Weighted-random top-N is a future personality knob — do not add it now.

## Global constraints (apply to every task)

Style, dependency discipline, and scope rules live in AGENTS.md — not repeated here. These are the refactor-specific constraints:

1. Do not change behavior of: opening build orders, combat micro execution, belief models — they are inputs, not targets. The scoring table module is in scope (Tasks 1-2); its rename may update import lines, comments, and structure listings in combat files — the fence is behavior, not file avoidance.
2. Every scoring change requires pure-Python unit tests pinning the scoring math BEFORE wiring it into the manager (test-first).
3. Every behavior change ships with its telemetry gate result; no regression vs master without a documented reason.
4. Preserve existing telemetry event names and fields. Add new ones freely; remove or rename nothing.
5. One task, one PR. PR description must state which acceptance criteria are met and which remain.
6. Every score follows the Design language: intel context only (no raw game-state access), every input through a response curve, re-score on cadence — never per frame.

## Task 1 — Rename + elevate scoring table

**Goal:** `target_scoring` module renamed to `scoring_table`; the `COUNTER_TABLE` dict becomes `SCORING_TABLE`. Its role documented as scoring core, not combat nudge.

**Scope:** module rename, import updates (6 sites — including the function-local import in `utilities/debug.py` that only executes in debug sessions, so a miss survives smoke runs), docstring stating the table's elevated role. `render_target_scoring_debug` keeps its name (it names the overlay). Comments, docstrings, and README/AGENTS structure listings referencing the old path update too.
**Preserve:** behavior — zero functional change.
**Acceptance:**
- [ ] old module path gone — `grep -rn "target_scoring" bot/` returns zero hits except the kept overlay function `render_target_scoring_debug` — and `scoring_table` imports resolve
- [ ] unit tests pass unchanged
- [ ] full roster smoke run matches master within noise

## Task 2 — Extend SCORING_TABLE to full Protoss arsenal

**Current:** 95 pairs, Liquipedia-annotated; attacker coverage extended in the 2026-10-19 pass (Log). Remaining gaps: mirror targets as enemy entries — Dark Templar, Disruptor, Mothership (0 rows) — plus a completion audit against `plan/counter_table_research_liquipedia.md`.

**Goal:** fill coverage gaps so any composition the bot builds or faces scores correctly. The scoring function can only compare compositions the table knows — blocking for the macro conversion (Task 4).

**Gaps to cover:**
- Missing enemy targets: Dark Templar, Disruptor, Mothership (mirror matchups)
- Audit remaining attacker/target rows for completeness vs the Liquipedia research plan

**Rules:** new pairs are Liquipedia-annotated, same format as existing entries.
**Acceptance:**
- [ ] every unit the bot can build has complete target rows (Observer exempt — no weapon)
- [ ] table passes a persisted format-validation test: no duplicate pairs, all values on one scale
- [x] no existing pair values modified

## Task 3 — Reasoner library + intel context

**Goal:** the shared foundation every decision system scores through — one library, not three bespoke scorers. Phase 2 adoption (combat, structures, expansion) reuses this module unchanged.

**Deliverables:**
- One shared module under `bot/utilities/`: response curves (linear / exponential / logistic), the appraisal type (base score, multiplier, veto), and the canonical combination runner — `score(choices) → ranked candidates`, deterministic argmax.
- `IntelContext`: the snapshot dataclass considerations score against — belief state, threat severity, intel urgency/freshness, economy, tech progress. Built by the intel layer on the re-score cadence, never per frame. Reads from the belief layer and intel quality systems; owns no state.
- Curve parameters load from config.

**Acceptance:**
- [ ] curve shapes + canonical combination pinned by unit tests (sum of bases, product of multipliers, any veto ⇒ 0)
- [ ] library imports nothing from manager code — dependency direction is managers → library
- [ ] Tasks 4-6 consume it; no manager defines its own curve or scorer

## Task 4 — Macro manager utility scoring

**Current:** `select_army_composition` — profile base → sticky econ switch → archon % threshold → nudges (scoring-table, strategy, resource-pressure, priority reorder), clamped + re-normalized.

**Goal:** composition candidates compete by score every macro cycle.

**Design (canonical form — see Design language):**
```
score(comp) = [ Σ counter_effectiveness(my_units vs believed_enemy_comp)
              + economy_term(worker count, income, saturation)
              + tech_term(upgrade/tech progress)
              − cost_term(minerals/gas/time) ] × belief_confidence
```
- The scoring table becomes the core base score (no longer a post-decision nudge).
- `belief_confidence` is a multiplier: expected utility, not an additive term.
- Candidates: current profile compositions + counter-driven variants + hold/defend candidate, capped at 8 per cycle. The winner builds.
- Remove sticky one-way switches and the threshold archon switch. Stabilizers: curve steepness first, commitment consideration second — never hard switches.
- Magnitudes: seed from Liquipedia baselines, fit later from telemetry (Task 8); document seed provenance in comments.

**Acceptance:**
- [ ] unit tests pin curve shapes and candidate scoring before integration
- [ ] econ switch removed, no one-way state remains
- [ ] branch passes full roster regression vs master in Compare tab

## Task 5 — Reaction manager utility scoring

**Current:** category routing (CHEESE/ALL_IN/TIMING/MACRO) behind confidence thresholds; handlers fire or don't. Firing rate on rush tapes: 0/6 (Log).

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

**Acceptance:**
- [ ] `used_cheese_response` fires on the decision-layer tape (cannon rush: 0/3 baseline, p_cheese 0.983) — the PR gate. Worker-rush firing (perception-layer: model classified MACRO 0.795; fix is model-side retraining, outside this plan) is reported but not gated. Survival-rate deltas reported for both tapes.
- [ ] `reactions` transition event emitted on every response activation and clear (category, level2 label, scores) — turns the binary end-state match field into a per-activation trace; timing/all-in responses become measurable, and "fired then cleared" is distinguishable from "never fired." New event, emission-on-change only; existing names and fields preserved (constraint 4).
- [ ] the scoring competition is subtype-agnostic by construction — the same `u(respond) vs u(continue)` decides worker rush, cannon rush, proxy, ling rush; the only subtype-specific data is each handler's own candidate cost/effect inputs. Verified by: one scoring path, N scored candidates.

## Task 6 — Scouting manager utility scoring

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

## Task 7 — Intel coordination wiring

**Goal:** single belief state read by all three managers through the intel layer.

**Design:**
- Managers register as intel consumers.
- Stale intel discounts contribution (decay), never zeroes it.
- Same `p_cheese` input feeds reaction and macro scores.
- A rising p_cheese must raise defense utility, shift composition, and raise scout priority together — all three read the same number.

**Acceptance:**
- [ ] no manager reads a private/duplicate belief source
- [ ] the four hard-coded `_intel_urgency > 0.5` gates (scout hunt, hallucination scout, expansion block, urgent observer) route through one shared staleness consideration — the urgent-observer gate lives in structure_manager and may follow in Phase 2
- [ ] staleness gate tested (decay curve, not binary cutoff)
- [ ] full regression vs master

## Task 8 — Fitted magnitudes from telemetry

**Goal:** reseed score weights using lab outcomes instead of Liquipedia seeds alone.

**Method:** fit per-term weights against the 2,300+ game corpus (e.g., immortal-vs-roach outcomes vs win rate).
**Acceptance:**
- [ ] fitted weights documented with fit evidence
- [ ] regression run shows ≥ no-worse across roster; improvements logged per matchup

## Adoption Roadmap (Phase 2-3 — not tasks in this plan)

Phase 1 proves the foundation on three managers. The remaining decision surfaces adopt the same reasoner in ROI order:

| # | System | Current mechanism | Phase |
|---|--------|-------------------|-------|
| 1 | Combat engagement — `handle_attack_toggles` engage/retreat | combat-sim gate + sticky duration timers | 2 |
| 2 | Chronoboost + Energy Recharge — target selection | first-match priority lists | 2 |
| 3 | Expansion — `_has_map_control` / `expansion_checker` | hard gates + threshold conjunctions | 2 |
| 4 | Defender allocation — threat response sizing | graduated severity table | 2 |
| 5 | Mass Recall — `try_mass_recall` gates | three AND-gates | 2 |
| 6 | Macro leftovers — tech-invest, gateway/forge counts, detection cannons, shield battery | booleans/thresholds | 2 |
| 7 | Per-unit micro — blink, storm, disruptor nova | thresholds | 3 |

Phase 3 needs a per-tier re-score cadence budget — per-unit scoring is the perf-sensitive case; the global N-loop cadence does not apply there.

## Verification (the test lab as harness)

- **Branch runs:** OpenCode works on the `development` branch; lab runs mount the branch worktree via the runner's `branch` param.
- **Unit-level:** pure-Python tests pinning the utility curves before refactor touches them (OpenCode writes these first).
- **Behavior-level:** the banked before-tapes re-run on the branch, each labeled with its failure layer:
  - Worker rush: PiG_Bot vs Who (WorkerRush) — 3/3 losses, 0/3 responses, belief peak p_macro 0.795. **Perception-layer failure:** the model classified MACRO and guards are deliberately fallback-only (model-authoritative design — guards fire only when the model is missing), so scoring never received a p_cheese to act on. Utility scoring cannot flip this tape; the fix is model-side (worker-rush features/training in the sklearn pipeline), owned by the belief layer — not guard rewiring. Report the delta; do not gate the PR on it.
  - Cannon rush: villain TBD (Chance/ProxyCannoneer proposed) — vs the constructed Who CannonRush, 3/3 wins, 0/3 responses, belief peak p_cheese 0.983 with zero action. **Decision-layer failure:** detection and confidence were right; routing failed to act. Task 5 owns this gate — target: action taken.
- **Macro lane:** at least one non-cheese series from the roster (e.g., MicroMachine) checked for composition-scoring stability — the refactor is bot-wide, not defined by the rush demo.
- **Regression:** full roster series (MicroMachine, Clicadinha, Who-random) master vs branch, compared in the lab's Compare tab.
- **Telemetry gates:** `used_cheese_response` firing rate, response latency, win-rate deltas per matchup. No regression vs master without a documented reason.


## Log

The plan stays forward-looking; verification history and progress notes are recorded here.

- **Pre-work:** Lab runner `branch` param confirmed — branch runs mount the `development` worktree.
- **2026-10-19 — Task 2 partial:** +34 rows added (61→95), 0 pair values changed vs the pre-#5 table; 0 duplicate keys, values −10.0..12.0. 8/9 buildable units have attacker rows (Observer exempt — no weapon). Remaining gap: mirror targets (Dark Templar, Disruptor, Mothership as enemy targets — 0 rows). Format-validation test added in 77dc34e, deleted in 0698240 per the run-then-delete convention — a persisted test is still owed.
- **2026-11-02 — v0.4 cleanup:** Scope fence clarified — out of scope is combat micro execution (runtime behavior); mechanical import/comment updates in combat files are in scope. Task 1 blast radius verified: 6 import sites; `render_target_scoring_debug` keeps its name. Principles merged into the Design language (each rule now stated once); historical annotations moved here from task bodies. Task 2 gap list refreshed from the live table: mirror targets (DT/Disruptor/Mothership as enemy rows, 0) remain; version-tagged firing-rate history (2/104 v0.13.1, 0/7 v0.13.2, 0/6 banked; p_cheese 0.983 no-action) moved here from Task 4.
- **2026-11-02 — v0.5 scope correction:** Goal reframed from "three managers" to bot-wide foundation with the managers as first adopters. New Task 3 (reasoner library + IntelContext) added ahead of the manager conversions; old Tasks 3-7 renumbered to 4-8 (task numbers in pre-0.5 log entries refer to the old numbering). Adoption Roadmap added for decision surfaces the original scope missed: combat engagement, chrono/recharge, expansion, defender allocation, recall gates, macro leftovers, per-unit micro. "Combat micro already utility-shaped" corrected — true only of `score_target()`; engagement, recall, and choke decisions remain threshold-based (Phase 2 material). Macro lane added to Verification so success is not defined by the rush tapes alone.
- **2026-11-02 — Tape relabeling:** Baselines section folded into the Verification tape bullets (numbers live next to their gates). Rush tapes labeled by failure layer: worker rush = perception (model classified MACRO 0.795 under the deliberate model-authoritative design — guards are fallback-only by intent, so scoring never saw a p_cheese; fix is model-side retraining, not guard rewiring); cannon rush = decision (p_cheese 0.983, zero action — Task 5's own gate). Task 5's firing gate re-scoped to the decision-layer tape; the worker-rush delta is reported but not PR-gating.
- **2026-11-02 — Task 5 acceptance generalized:** response scoring is subtype-agnostic by construction (same `u(respond) vs u(continue)` for all cheese subtypes; handlers as scored candidates) — now pinned by acceptance instead of intent. Added the `reactions` activation/clear transition event so firing is measurable per-activation across ALL_IN/TIMING too, and "fired then cleared" is distinguishable from "never fired" (the binary `used_cheese_response` match field stays as-is per constraint 4). Cannon-rush tape remains the PR gate instance, not the definition of success.
