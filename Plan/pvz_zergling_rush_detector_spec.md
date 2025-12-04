# PvZ Zergling Rush Detector — Detection-Only Spec

## Scope
Detect Zergling pressure in PvZ and classify into one of **three labels**. No defense logic, no travel-time offsets.

---

## Labels

| Label | Definition |
|-------|------------|
| **12_pool** | Very early spawning pool (~12 supply), producing the earliest Zergling rush |
| **speedling** | Early pool + early gas leading to a fast Metabolic Boost timing rush |
| **macro** | Normal/standard opener (negative class) |

---

## Timing Constants (Faster game seconds)

| Name | Time | Purpose |
|---|---:|---|
| `T_POOL_12P_START_MIN` | **38.0** (≈0:38) | 12-pool start window min |
| `T_POOL_12P_START_MAX` | **42.0** (≈0:42) | 12-pool start window max |
| `T_POOL_12P_DONE` | **65.0** (≈1:05) | 12-pool completion |
| `T_POOL_SPEED_START_MIN` | **48.0** (≈0:48) | Speedling pool start min |
| `T_POOL_SPEED_START_MAX` | **52.0** (≈0:52) | Speedling pool start max |
| `T_POOL_MACRO_MIN` | **52.0** (≈0:52) | Macro pool start min |
| `T_POOL_MACRO_MAX` | **80.0** (≈1:20) | Macro pool start max |
| `T_NAT_CHECK` | **80.0** (≈1:20) | Natural hatch must exist in macro |
| `T_LING_12P_SEEN` | **115.0** (≈1:55) | 12-pool lings seen threshold |
| `T_LING_SPEED_SLOW` | **135.0** (≈2:15) | Speedling slow-lings appear |
| `T_SPEED_START_EARLY` | **85.0** (≈1:25) | Early speed research start |
| `T_SPEED_HIT` | **160.0** (≈2:40) | Speedling hit window |
| `T_QUEEN_12P_CHECK` | **120.0** (≈2:00) | No queen by this time = 12-pool signal |
| `T_QUEEN_NORMAL` | **120.0** (≈2:00) | Normal queen start time |

**Late-scout reconstruction:**
```python
start_time = self.bot.time - building.build_progress * building._type_data.cost.time / 22.4
```

---

## Auto-TRUE Guards (evaluate before scoring)

| Guard | Condition | Result |
|---|---|---|
| A | Any Zergling **seen ≤ 1:45** | **Return `12_pool`** |
| B | Slow-ling **contacts nat ≤ 2:00** | **Return `12_pool`** |
| C | Speed-ling **contacts nat ≤ 2:40** | **Return `speedling`** |

If none fire, proceed to scoring.

---

## Signals: `12_pool`

| Signal | Condition | Weight |
|---|---|---:|
| Early pool start | Pool start ≈ **0:38–0:42** | +4 |
| Pool done early | Pool **done ≤ 1:05** | +3 |
| No natural | Natural absent at **≥ 1:20** | +3 |
| Very early lings | Lings **seen ≤ 1:55** | +4 |
| No queen | No queen by **2:00** | +2 |

**Notes:**
- Gas may be present or absent (does NOT change the label)
- Optional features (for ML): early extractor, baneling nest timing, gas mined

---

## Signals: `speedling`

| Signal | Condition | Weight |
|---|---|---:|
| Speedling pool timing | Pool start ≈ **0:48–0:52** | +3 |
| Early gas | Extractor taken early (≈12 gas) with workers mining | +3 |
| Speed research early | Speed starts **≤ 1:25** | +4 |
| Slow lings appear | Lings seen around **2:05–2:15** | +2 |
| Speed-ling hit | Speedlings contact **≤ 2:40** | +4 |
| Queen timing normal | Queen started around **1:50–2:00** | −1 (damper) |
| Natural on time | Natural hatch appears **≤ 1:20** | −1 (damper) |

**Notes:**
- Natural may appear on time (disguise element)
- Pressure comes from speed timing, not raw arrival timing
- Optional features: baneling nest 2:00–2:20, gas cut at 100

---

## Signals: `macro` (negative class)

| Signal | Condition | Weight |
|---|---|---:|
| Normal pool timing | Pool start **0:52–1:20** | +2 |
| Normal natural | Natural taken **1:00–1:20** | +2 |
| Normal speed timing | Speed starts **≥ 1:30** | +2 |
| Lings appear late | Lings seen **> 2:10** | +2 |
| Normal queen | Queen starts **1:50–2:00** | +1 |

---

## Classification Logic

```python
# Auto-TRUE guards
if first_ling_seen <= 105.0:  # 1:45
    return "12_pool"
if first_ling_contact <= 120.0 and not ling_has_speed:  # 2:00 slow-ling
    return "12_pool"
if first_ling_contact <= 160.0 and ling_has_speed:  # 2:40 speed-ling
    return "speedling"

# Scoring
score_12p = compute_12pool_score(signals)
score_speed = compute_speedling_score(signals)
score_macro = compute_macro_score(signals)

# Classification
if score_12p >= 5:
    return "12_pool"
elif score_speed >= 5:
    return "speedling"
else:
    return "macro"
```

---

## Feature Inputs (detector expects)

- **Times:** `time_now`, `pool_start_time` (estimated), `nat_start_time` (estimated)
- **Structures:** pool object, natural hatch status, extractor status, queen status
- **Unit sightings:** first ling sight time, first contact time, ling has speed flag
- **Resource reads:** workers on gas (if visible)

---

## Evaluation Order

1. Check **Auto-TRUE** guards A/B/C → early return if any fires
2. Build features; compute `start_time` from build progress if needed
3. Score signals for `12_pool`, `speedling`, `macro`
4. Return highest-scoring label (or `macro` if both rush scores < 5)

---

## Logging (detection only)

Minimal row per game:
```
map_name, rush_distance_seconds,
t_nat_started(est), pool_seen_state, t_pool_start(est),
t_gas_seen, t_queen_started, t_speed_research,
t_ling_seen, t_ling_contact, ling_has_speed,
score_12p, score_speed, rush_label,
result, game_time_seconds
```
