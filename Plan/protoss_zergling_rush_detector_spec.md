# Protoss Zergling-Rush Detector, Map-Aware, Boolean Output

**Purpose.** Decide if the opponent is executing an early Zergling rush. Return a boolean your planners can trust. Internally, use a transparent score built from observable facts, aligned to a per-map rush clock.

**Interface.** Keep your public API unchanged, ARES-style:

```python
def get_enemy_ling_rushed(state) -> bool:
    ...
```

Everything below is the internal logic. Use your existing variable names where you already have them. Names here are suggestions only.

---

## Core Idea

Infer intent by comparing current observations to what a normal Hatch-first looks like at specific time checkpoints. Those checkpoints are shifted by your per-map rush time so the same logic holds on short, medium, or long maps. Early Pool, missing natural, early lings, and on-time contact are the strongest signals. Gas and Queen timings refine, they never decide alone.

---

## Inputs the Detector Must Have

Use your existing names where available.

- `time_now` in game seconds, Faster.
- `rush_time_zergling_seconds` from your Dijkstra module, **ramp-to-ramp** slow Zergling travel time.  
  If your module returns tiles, convert once with `4.13` tiles per second.

Enemy state snapshots:
- Natural: `enemy_nat_started_at: float | None`
- Spawning Pool:
  - `pool_seen_state ∈ {none, morphing, done, unknown}`
  - `pool_seen_time: float | None`
- Extractor: `extractor_seen_time: float | None`
- First Queen: `queen_started_time: float | None`
- Zerglings:
  - `first_ling_seen_time: float | None`
  - `first_ling_contact_nat_time: float | None`

---

## Map Offset, What It Is, Why It Matters

Your rush-time module already computes how long a slow Zergling needs to go from their ramp to yours. Use that to shift every checkpoint.

- If you store tiles:
  ```python
  rush_time_zergling_seconds = path_len_tiles / 4.13
  ```
- Define once at game start:
  ```python
  map_offset = rush_time_zergling_seconds - 120.0
  ```
  The 120.0 seconds reference is the nominal 12-Pool slow-ling contact on a medium rush distance.  
  Positive on long maps, negative on short maps.

---

## Derived Checkpoints, Protoss Specific

Compute once per game, then reuse.

```python
T_POOL_CHECK        = 65.0  + map_offset   # ≈ 1:05
T_NAT_CHECK         = 80.0  + map_offset   # ≈ 1:20
T_LING_SIGHT        = 105.0 + map_offset   # ≈ 1:45
T_LING_CONTACT_NAT  = 120.0 + map_offset   # ≈ 2:00
T_QUEEN_CHECK       = 125.0 + map_offset   # ≈ 2:05
```

These are conservative and shift cleanly with map length.

---

## Calculating Building Start Times from Partial Information

When you see a structure already in progress, estimate its start time. Use your provided formula verbatim:

```python
start_time = self.bot.time - building.build_progress * building._type_data.cost.time / 22.4
```

- `self.bot.time` is current game time.  
- `building.build_progress` in [0, 1].  
- `building._type_data.cost.time` is the build duration.  
- Division by `22.4` converts game loops to Faster-speed seconds so the subtraction lands in seconds.  

Use this for Pool timing checks when your scout arrives mid-build. For example, if estimated `start_time + pool_build_time` implies “Pool done by T_POOL_CHECK,” set `pool_seen_state` accordingly.

---

## Evidence Features and Why They Matter

All checks are evaluated against the derived checkpoints above.

### 1) No natural by `T_NAT_CHECK`  (strong)
- **Definition**
  ```python
  no_natural_by_T = (enemy_nat_started_at is None) and (time_now >= T_NAT_CHECK)
  ```
- **Why**  
  In macro, Zerg commits minerals to the natural early. Skipping it strongly implies minerals went to Pool and Lings.
- **Use**  
  Loud non-combat tell. High weight.

### 2) Pool early with no natural  (strong)
- **Definition**
  ```python
  pool_early = (
      (pool_seen_state in {"morphing", "done"}) and
      (pool_seen_time is not None) and
      (pool_seen_time <= T_POOL_CHECK) and
      (enemy_nat_started_at is None)
  )
  ```
  If you scouted mid-build, estimate with the start-time formula above.
- **Why**  
  Early Pool plus missing natural is the classic 12-Pool family signature.
- **Use**  
  High precision when main vision exists. If main was never seen, keep this “unknown,” do not assume.

### 3) Early lings on the map  (strong)
- **Definition**
  ```python
  early_lings_seen = (
      (first_ling_seen_time is not None) and
      (first_ling_seen_time <= T_LING_SIGHT)
  )
  ```
- **Why**  
  Macro Zerg does not float early Lings this soon because it costs drones. Seeing them early is a behavioral tell.
- **Use**  
  Strong signal that is robust to fog in the main.

### 4) Ling contact by `T_LING_CONTACT_NAT`  (very strong, confirmatory)
- **Definition**
  ```python
  ling_contact_by_T = (
      (first_ling_contact_nat_time is not None) and
      (first_ling_contact_nat_time <= T_LING_CONTACT_NAT)
  )
  ```
- **Why**  
  This is the arrival window implied by your map-aware rush time. Contact at or before confirms a rush.
- **Use**  
  Confirmation. If contact does not occur by this time, down-weight panic unless other strong evidence exists.

### 5) Gasless early  (refiner)
- **Definition**
  ```python
  gasless_early = (extractor_seen_time is None) and (time_now >= 80.0 + map_offset)
  ```
- **Why**  
  No gas means minerals are likely going into slow-ling pressure instead of tech. Helps separate immediate slow-ling hits from delayed speed floods.
- **Use**  
  Refinement only. Do not decide alone.

### 6) Queen late while lings are out  (refiner)
- **Definition**
  ```python
  queen_late_with_lings = (queen_started_time is None) and early_lings_seen and (time_now >= T_QUEEN_CHECK)
  ```
- **Why**  
  Queens are the macro stabilizer. Delaying the first Queen while producing Lings signals a bigger commitment to pressure.
- **Use**  
  Refinement only.

---

## Scoring Model and Thresholds

Use small integers. Strong signals get 3 or 4. Refiners get 1. This keeps decisions explainable.

```python
score = 0

# strong signals
if no_natural_by_T:            score += 3
if pool_early:                 score += 3
if early_lings_seen:           score += 3
if ling_contact_by_T:          score += 4   # confirmation

# refiners
if gasless_early and early_lings_seen:  score += 1
if queen_late_with_lings:               score += 1
```

**Classification label** for debugging or dashboards. Keep your boolean API the same.

- `score ≥ 6`  → `rush_label = "12pool_class_rush"`
- `3 ≤ score ≤ 5` → `rush_label = "pool_first_pressure"`
- `score ≤ 2`  → `rush_label = "likely_macro"`

**Boolean return** for your current interface:

- Recommended `return (score >= 5)`  
  This flips to true on either confirmation alone, or any two strong signals, or one strong plus two refiners. If your current threshold differs, keep yours and map it to `score`.

**Debounce rule** to avoid flapping:

- Once `rush_label == "12pool_class_rush"`, hold it for 20 game seconds unless a natural appears and there is no Ling contact by `T_LING_CONTACT_NAT + 10`.

---

## False Positive Reducers

- If a natural appears within 10 to 15 seconds after `T_NAT_CHECK` and there are no Lings by `T_LING_SIGHT`, subtract 1 from `score`. This is a defensive pool-first, not a rush.
- If Speed is seen started early but no Lings by contact time, expect a later flood. Tighten wall and get a second unit, but do not escalate to “rush” unless another strong signal appears.

---

## What This Means for Protoss Defense Logic

The detector does not issue actions. It sets intent for your planner.

- If boolean is true or `rush_label == "12pool_class_rush"`  
  Complete the wall at the natural. Chrono the first Zealot. Pull 2 to 4 probes to hold the gap if needed. Delay Nexus if wall is threatened. Keep backup power to the wall.
- If `rush_label == "pool_first_pressure"`  
  Keep wall ready, Zealot on hold at the gap. Nexus timing is allowed once your first unit spawns and there is no contact by the window. Re-scout the natural and downgrade if it appears quickly.

---

## Logging, Why It Exists, How It Improves Results

Yes, the logger exists so the detector gets better over time and to unlock ML later. Keep it minimal and numeric. One row per game. Use your names if you already have them.

Suggested fields:

```json
{
  "map_name": "RoyalBloodAIE",
  "rush_time_zergling_seconds": 118.8,
  "map_offset": -1.2,

  "t_nat_started": null,
  "pool_seen_state": "done",
  "t_pool_seen": 63.1,
  "t_extractor_seen": null,
  "t_queen_started": null,

  "t_first_ling_seen": 100.5,
  "t_first_ling_contact_nat": 117.4,

  "score": 8,
  "rush_label": "12pool_class_rush",
  "structs_lost_first3": 1,
  "workers_lost_first3": 5,
  "win": false
}
```

Immediate payoff:
- Compute per-map medians for `t_first_ling_contact_nat` and replace `map_offset` with empirical offsets where you have data. This tightens thresholds and cuts false alarms.

Later, light ML:
- **Supervised calibration.** Train a small classifier to predict the three labels from the features above. Start with logistic regression or a shallow tree. Keep the boolean threshold as a guardrail.
- **Policy tuning.** Independently of detection, analyze defense choices versus early outcomes to optimize probe pull count and second-unit selection.

---

## Failure Modes and How This Design Handles Them

- **No main-base vision.** Pool state remains “unknown.” Detector still works on natural timing and Ling timing. Score will not overreact on one refiner alone.
- **Scouting denial at the natural.** If you miss the natural moment, early Lings and the contact window still drive the decision.
- **Long maps and chronically late naturals.** Map offset shifts checkpoints later. The false-positive reducer demotes cases where the natural appears right after the check and there are no Lings.
- **Delayed speedling floods.** Detector does not flip to “rush” at 2:00 unless other strong signals occur. Your planner still tightens the wall when Speed is seen early.

---

## Integration Notes

- Keep your public method exactly as it is:
  ```python
  def get_enemy_ling_rushed(state) -> bool:
      # compute map_offset from your rush-time module
      # estimate building start times when needed
      # compute features, score, and boolean as above
      # do not rename existing variables
  ```
- If your rush-time module currently outputs worker time rather than Zergling time, convert once:
  ```python
  rush_time_zergling_seconds = rush_time_worker_seconds * (worker_speed / 4.13)
  ```
  Use your worker speed constant if you already store it.

---

**End of spec.**
