---
project: PiG_Bot
version: 1.0
author: Drekken
goal: "Develop a human-like Protoss bot following PiG's B2GM 2023 double-robo macro style."
framework: ARES
language: Python
status: active
priority: high
modules:
  - Combat
  - Tactics
  - Macro
  - Build Manager
  - Scouting
  - Logging & Data Storage
  - Analysis & Learning
  - Replay Parsing
dependencies:
  - python-sc2
  - ARES
  - pandas
  - matplotlib
  - sc2reader
tags:
  - StarCraft II
  - AI
  - Bot Development
  - PiG_Bot
  - Windsurf
---

# PiG_Bot –  Project Plan 

## Objectives
- Build a Protoss SC2 bot that plays like a human (Bronze → GM) using PiG’s B2GM robo‑centric style.
- Implement clear modules for Macro, Combat, Scouting, Build Management, and Reactive Defense.
- Support difficulty control, pre‑defined build switches, and basic micro (stutter‑step, disruptors).
- Log per‑match data for analysis and future lightweight adaptation.

## Scope: What the Bot Should Do
- Human‑like play patterns; difficulty control; adaptive reactions.
- Behaviors: build Bronze→GM macro style; scout and interpret; defend cheese (zergling/cannon/probe rush);
  disruptor usage; basic stutter‑step; respond to air; stay near 3 bases/66 probes; pre‑defined build switches;
  fight only when favored; two builds per matchup (PvP 4‑Gate All‑in / 2‑Gate Expand; PvZ Standard Robo / Double SG Phoenix; PvT Standard Macro / Proxy Rax Response).

---

## Tasks

### Combat ✅ (from original **Plan → Combat**)
- [x] Integrate Combat simulator `can_win_fight`  
  - [x] Only attack if enemy combat score == 0
- [x] Create Probe Rush response
- [x] Enhance Probe Rush response with `WorkerKiteBack` and `DEFENDING` role (replicate cannon rush response)
- [x] Test Cannon Rush Response
- [x] Fix Cannon Rush Response
- [x] Enhance Observers  
  - [x] Army‑observer path distance  
  - [x] Multi‑state (Follow/Reveal)  
  - [x] Move to unseen enemies  
  - [x] Add additional observers  
  - [x] Surveillance mode  
  - [x] Split into SCOUT & FOLLOW roles
- [x] Send Zealot into the wall hole
  - [x] Fix units blocked by gatekeeper zealot
  - [x] Rally units inside, not outside
  - [x] Dodge ravager/disruptor balls & nukes
  - [x] React to Liberator zones
  - [x] React to invisible enemies

### Tactics ✅
- [x] Break army into squads
- [x] Per‑squad attack targets & combat simulators
- [x] Anti‑air squad (only AA units; proportional to threat)

### Macro
- [ ] Build‑order recovery if buildings get destroyed (consult Rasper)
- [ ] Adjust PvP strategy to 1‑base (instead of natural)
- [ ] Fluid weighted composition vs enemy comp

### Build Manager
- [ ] Implement PvT Standard Macro build
- [ ] Implement PvT Proxy Rax Response build
- [ ] Implement PvZ Standard Robo Opener
- [ ] Implement PvZ Double Stargate Phoenix (vs Mutas)
- [ ] Implement PvP 4‑Gate All‑in
- [ ] Implement PvP 2‑Gate Expand
- [ ] Build selector by opponent race + scout info
- [ ] Connect selected build to `BuildOrderRunner` at game start
- [ ] Mid‑game build switch logic (e.g., Spire → Stargate)
- [ ] Tests for builds (ideal + edge conditions)

### Scouting & Interpretation
- [ ] `scouted_spire()` and other scout triggers
- [ ] Fallback logic if scout dies early
- [ ] Interpret scout info → build decisions
- [ ] Observers ensure mid‑game critical tech scouting

### Logging & Data Storage
- [ ] Log per‑game JSON to `/logs/`  
  - opponent_race, map_name, game_result, build_order_used, build_loss_flag, enemy_units_seen, biggest_fight_outcome
- [ ] Use pandas‑friendly JSONL formatting
- [ ] `on_end()` post‑game hook triggers logging

### Analysis & Learning (Post‑Match)
- [ ] Analyze performance across matchups (pandas)
- [ ] Chart winrates over time (matplotlib)
- [ ] Visualize build performance by race/opponent/map
- [ ] Lightweight build adaptation (e.g., avoid low‑winrate builds)

### Replay Parsing (Optional)
- [ ] Test `sc2reader` on bot & human replays
- [ ] Extract build order + APM
- [ ] Compare build execution vs human timing
- [ ] Store replay summaries alongside JSON logs

### Misc / TODO
- [x] Change Gamestate from text → INT
- [x] Optimal worker check includes minerals + gas
- [ ] Review macro cycle (floating minerals during build order)
- [ ] Worker‑rush returns to mining afterward
- [ ] Add cheese detection: structure near our town hall / mediator `get_enemy_PF_rushed`
- [ ] Adjust `Worker_rush` detection range
- [x] Remove Target Unit; use Target Unit **position**
- [x] Check if regrouping is too aggressive in `regroup_army`
- [ ] Check threat sensitivity—Marines/Marauders/Reapers
- [ ] Review Battle_Sim sensitivity + `all_out_attack`
- [ ] Add `Did_Enemy_rush` hook
- [x] Remove Overcharge
- [ ] Test Energy Recharge
- [ ] Test new battle‑sim conditions
- [ ] Fix battle lockout bouncing
- [x] Cap workers at a set limit (80; 8 on gold base)
- [ ] Add upgrades logic
- [ ] Inner/outer radius for unit grouping; break when 30% engaged
- [x] Fix Gatekeeper off in non‑PvZ

### Gatekeeper System (PvZ Wall Logic) ✅
- [x] Issue Hold Position
- [x] Move to zealot hole in wall  
  - [x] Identify the hole
- [x] If no zealot in role, add zealot Gatekeeper role
- [x] Create Gatekeeper role
- [x] Ensure only active in PvZ
- [x] Only 1 zealot at a time
- [x] Hold position unless attacking
- [x] Release role when attacking, **not** when defending


---

## Data To Log (schema)
- `opponent_race`, `map_name`, `game_result`, `build_order_used`, `build_loss_flag`, `time_build_switched`,
  `enemy_opening_seen`, `enemy_first_army_unit`, `enemy_air_tech_seen`, `biggest_fight_outcome`, `units_lost_total`,
  `enemy_units_seen`, `scouted_before_4min`, `reacted_to_cheese`, `num_expansions_taken`.

## Dependencies
- `python-sc2`, `ARES`, `pandas`, `matplotlib`, `JSON` (logs), `sc2reader` (optional)

## Future Goals (After MVP)
- Difficulty selection; mode types (macro/harass/reactive); adaptive build switching via scoring/scouting; 4+ builds per MU; advanced micro (FF, blink, AoE splits); terrain logic for more maps; play other races; disable speed mining.

## Notes / Observations
- Could rally at the natural entrance instead of natural base.
- 17 marines didn’t trip threat.
- Drawn to injured units? Investigate targeting.
- Units rally toward expansion; may be better to rally 2 units in front of natural toward center.
