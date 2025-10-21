# PvZ Zergling Rush Detector — Detection-Only Spec

## Scope
Detect **three** Zergling rushes in PvZ and return a boolean. No defense logic, no travel-time, no offsets.

- **R1:** 12-Pool, Gasless  
- **R2:** 12-Pool, 11-Gas (ling-bane / early-speed from 12-pool)  
- **R3:** 13/12 Speedling

---

## Global Detection Clocks (Faster game seconds)
| Name | Time | Purpose |
|---|---:|---|
| `T_POOL_CHECK` | **65.0** (≈1:05) | Earliest pool completion check |
| `T_NAT_CHECK` | **80.0** (≈1:20) | Natural hatch must exist in macro |
| `T_LING_EARLY` | **105.0** (≈1:45) | Early ling sighting, auto-TRUE |
| `T_LING_SOFT` | **110.0** (≈1:50) | Early lings, strong points only |
| `T_CONTACT_SLOW` | **120.0** (≈2:00) | Slow-ling contact, auto-TRUE |
| `T_CONTACT_FAST` | **150.0** (≈2:30) | Speed-ling contact, auto-TRUE |
| `T_QUEEN_CHECK` | **125.0** (≈2:05) | Queen sanity with early lings |

**Late-scout reconstruction (use if you arrive after a gate):**
```python
start_time = self.bot.time - building.build_progress * building._type_data.cost.time / 22.4
```
Use reconstructed `pool_start_time` / `nat_start_time` against the same clocks.

---

## Auto-TRUE Guards (evaluate before scoring)
| Guard | Condition | Result |
|---|---|---|
| A, Early Ling Sighting | Any Zergling **seen ≤ 1:45** | **Return True** |
| B, Slow-Ling Contact | First ling **contacts wall/nat ≤ 2:00** | **Return True** |
| C, Speed-Ling Contact | First **speedling** contact **≤ 2:30** | **Return True** |

If none fire, proceed to scoring.

---

## Signals by Rush Type (with weights)

### R1, 12-Pool Gasless
| Signal | Condition | Weight |
|---|---|---:|
| No natural | Natural absent at **≥1:20** (or reconstructed start after 1:20) | +3 |
| Early pool | Pool **done ≤1:05** and no natural, or reconstructed start ≈ 0:18–0:25 | +3 |
| No gas | No extractor by **1:20** | +2 |
| No queen | Spawning pool visible but no queen by **2:05** | +1 |
| Early lings | Any ling **seen ≤1:50** (if ≤1:45, Auto-TRUE A already fired) | +4 |
| Damper (v2) | Natural appears **≤1:35** and **no** lings by **1:50** | −1 |

**Label for logs:** `12p_gasless`

---

### R2, 12-Pool 11-Gas (ling-bane / early-speed from 12-pool)
| Signal | Condition | Weight |
|---|---|---:|
| No natural | Natural absent at **≥1:20** | +3 |
| Early pool | Pool **done ≤1:05** (or reconstructed early) | +3 |
| Active gas | Extractor up and **mined** (~3 workers) by ~**1:10** | +2 |
| Gas→100 cut | Gas bank ≈ **100**, then workers pulled (speed-only tell) | +2 |
| Speed research | Pool “wiggle” (Metabolic Boost) **~1:40–1:50** | +2 |
| Early lings | Ling **seen ≤1:50** | +3 |
| Baneling nest | Nest visible **2:00–2:20** | +2 |
| No queen | Spawning pool visible but no queen by **2:05** | +1 |

**Label for logs:** `12p_11gas`

---

### R3, 13/12 Speedling
| Signal | Condition | Weight |
|---|---|---:|
| 13 pool + early gas | Pool start ≈ **0:35–0:40** with early gas actively mined | +3 |
| No natural | Natural absent at **≥1:20** (or token/very late) | +2 |
| Gas→100 cut | Gas bank ≈ **100**, then workers pulled | +2 |
| Speed research | Pool “wiggle” **~1:40–1:50** | +2 |
| Speed contact | Speedlings **contact ≤2:30** (Auto-TRUE C would also catch) | +4 |
| Early lings | Ling **seen ≤1:50** | +3 |
| No queen | Spawning pool visible but no queen by **2:05** | +1 |
| Damper (v2) | Natural **≤1:35** and **no** speed signs by **2:00** | −1 |

**Label for logs:** `13_12_speed`

---

## Scoring → Boolean
- Sum all applicable signal weights (from any rush type).  
- **Threshold:** `is_rushed = (score >= 5)` unless an **Auto-TRUE** already returned `True`.  
- Optional telemetry: the **highest-scoring** rush type becomes `rush_label`.

---

## Feature Inputs (detector expects)
- Times, Faster seconds: `time_now`, `t_probe_main`, `t_probe_nat`  
- Structures: pool object, natural hatch status, extractor status, baneling nest status, queen status  
- Unit sightings: first ling sight time, first contact time, ling had speed flag  
- Resource reads (if available): gas mined amount, workers on gas

---

## Evaluation Order (pseudoflow)
1) Check **Auto-TRUE** A/B/C → early return `True` if any.  
2) Build features; if past a checkpoint, compute **start_time** from build progress.  
3) Score signals across R1, R2, R3.  
4) `return (score >= 5)`; set `rush_label` to max-score class for logs.

---

## Logging (detection only)
Minimal row per game:
```
map_name, rush_distance_seconds,
t_probe_nat, t_probe_main,
t_nat_started(est), pool_seen_state, t_pool_seen, pool_start_est,
t_gas_seen, gas_mined_est, t_queen_started,
t_ling_seen, t_ling_contact, ling_has_speed,
score, rush_label, is_rushed,
result, game_time_seconds
```
