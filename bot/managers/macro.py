from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId

from cython_extensions import cy_upgrade_pending, cy_unit_pending, cy_structure_pending_ares

from sc2.position import Point2
from sc2.units import Units

from ares.behaviors.macro import (
    ProductionController,
    SpawnController,
    MacroPlan,
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    UpgradeController,
    BuildStructure,
)
from ares.consts import UnitRole, WORKER_TYPES
from ares.consts import LOSS_MARGINAL_OR_BETTER

from bot.utilities.performance_monitor import get_economy_state
from bot.utilities.intel import get_enemy_intel_quality
from bot.combat.target_scoring import COUNTER_TABLE
from bot.constants import (
    MEMORY_EXPIRY_TIME,
    STALE_INTEL_THRESHOLD,
    RESOURCE_PRESSURE_MAX_NUDGE,
    RESOURCE_IMBALANCE_RATIO,
    FREEFLOW_INCOME_RATIO_THRESHOLD,
)
from ares.dicts.cost_dict import COST_DICT


def get_freeflow_mode(bot) -> bool:
    """
    Freeflow calculation: detects when SpawnController should `continue` past
    unaffordable units instead of `break`ing the loop.
    
    Uses a layered approach:
      1. Early-game pressure checks (one base, PvP)
      2. Bank-imbalance checks (always active, no PM dependency)
      3. Income-aware checks (when PerformanceMonitor data is available)
    
    Args:
        bot: The bot instance
        
    Returns:
        bool: True if should use freeflow mode
    """
    minerals = bot.minerals
    vespene = bot.vespene
    
    # --- Early game pressure ---
    if len(bot.townhalls) == 1 and minerals >= 280 and vespene >= 105:
        return True
    
    if (bot.enemy_race == Race.Protoss
        and len(bot.townhalls) <= 1
        and cy_unit_pending(bot, UnitTypeId.IMMORTAL)):
        return True
    
    # --- Bank-imbalance checks (always active, catches resource skew) ---
    # Both resources high — just spend
    if minerals > 500 and vespene > 500:
        return True
    
    # Significant bank with clear imbalance — one resource is 3x+ the other
    # and the dominant resource exceeds 400 (enough that we should be spending it)
    if minerals > 400 and minerals > 3 * max(vespene, 1):
        return True
    if vespene > 400 and vespene > 3 * max(minerals, 1):
        return True
    
    # --- Income-aware checks (more precise, needs PM data) ---
    pm = getattr(bot, 'performance_monitor', None)
    if pm and pm.tracking_enabled and pm.samples_taken >= 5:
        income_min = pm.avg_income_minerals
        income_gas = pm.avg_income_vespene
        
        # Severe income imbalance: one resource accumulates faster than we can
        # spend it. Trigger freeflow to keep production flowing with whatever
        # resources we DO have. No bank threshold — the imbalance itself is
        # the signal, and our Layer 1 priority reorder handles what gets built.
        if income_gas > 0 and income_min / (income_gas + 1) > FREEFLOW_INCOME_RATIO_THRESHOLD:
            return True
        if income_min > 0 and income_gas / (income_min + 1) > FREEFLOW_INCOME_RATIO_THRESHOLD:
            return True
    
    # Normal proportional production
    return False


def get_optimal_gas_workers(bot) -> int:
    """
    Dynamically adjust gas workers based on economy state.
    Prevents gas flooding and mineral starvation.
    
    Args:
        bot: The bot instance
        
    Returns:
        int: Number of workers per gas (0-3)
    """
    # Get current gatherers (workers actually mining)
    gatherers = bot.mediator.get_units_from_role(role=UnitRole.GATHERING)
    
    # Critical mineral starvation - stop gas completely
    if (bot.minerals < 100 
        and bot.vespene > 300 
        and bot.supply_used < 84):
        return 0
    
    # Early game with few workers - stop gas temporarily
    if ((bot._used_cheese_response and len(gatherers) < 21)
        or len(gatherers) < 12):
        return 0
    
    # Vespene flooding - toggle off gas mining
    if bot._gas_worker_toggle and bot.vespene > 1500 and bot.minerals < 100:
        bot._gas_worker_toggle = False
        return 0
    
    # Vespene starved - toggle gas mining back on
    if bot.vespene < 400 or bot.minerals > 1200:
        bot._gas_worker_toggle = True
    
    # Low gas - max workers
    if bot.vespene < 100:
        return 3
    
    return 3 if bot._gas_worker_toggle else 0

# Army composition constants 
STANDARD_ARMY_0 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 4},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.4, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 5},
    UnitTypeId.STALKER: {"proportion": 0.0, "priority": 2},
    UnitTypeId.ZEALOT: {"proportion": 0.2, "priority": 3},
}
STANDARD_ARMY_1 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.35, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.2, "priority": 4},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 3},
    UnitTypeId.STALKER: {"proportion": 0.0, "priority": 2},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 0},
}

PVP_ARMY_0 = {
    UnitTypeId.DISRUPTOR: {"proportion": 0.2, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 3},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.5, "priority": 0},
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 2},
}

PVP_ARMY_1 = {
    UnitTypeId.DISRUPTOR: {"proportion": 0.25, "priority": 0},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 2},
    UnitTypeId.IMMORTAL: {"proportion": 0.6, "priority": 1},
}

CHEESE_DEFENSE_ARMY = {
    UnitTypeId.ADEPT: {"proportion": 0.25, "priority": 2},
    UnitTypeId.STALKER: {"proportion": 0.15, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.6, "priority": 1},
    
    
}

# ===== PRODUCTION NUDGING =====
# Uses COUNTER_TABLE to shift army proportions toward unit types that are
# effective against the observed enemy composition. See COUNTER_TABLE docstring
# in target_scoring.py for how to extend.
#
# HOW IT WORKS:
#   1. Read cached enemy army (filtered: no workers, structures, expired ghosts)
#   2. For each unit type in our composition, sum its COUNTER_TABLE bonuses
#      against all observed enemy units → "effectiveness score"
#   3. Normalize effectiveness scores to a [-MAX_NUDGE, +MAX_NUDGE] range
#   4. Add nudges to base proportions, clamp to MIN_PROPORTION, re-normalize to 1.0
#
# HOW TO EXTEND:
#   - Adding a new unit to an army composition dict? Add COUNTER_TABLE entries
#     in target_scoring.py. The nudging picks them up automatically.
#   - Adding a new army composition dict? Pass it through select_army_composition()
#     and nudging applies to it like any other.

PRODUCTION_MAX_NUDGE = 0.10
"""Maximum proportion shift per unit type (±10%). Keeps changes conservative."""

PRODUCTION_MIN_PROPORTION = 0.05
"""Floor: never drop a unit type below 5% proportion (prevents starvation)."""

PRODUCTION_MIN_ENEMY_UNITS = 3
"""Don't nudge if fewer than this many enemy combat units seen (too little data)."""


ANTI_COLOSSUS_TYPES = {
    UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT, UnitTypeId.CORRUPTOR,
    UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.TEMPEST,
}
"""Enemy unit types that hard-counter Colossus. Drives nudging in all matchups."""


def _count_enemy_anti_air(bot) -> int:
    """Count enemy anti-Colossus air units from cached intel."""
    return sum(1 for u in _get_enemy_combat_units(bot) if u.type_id in ANTI_COLOSSUS_TYPES)


def _count_enemy_carriers(bot) -> int:
    """Count enemy Carriers from cached intel. Used for PvP Storm trigger."""
    return sum(1 for u in _get_enemy_combat_units(bot) if u.type_id == UnitTypeId.CARRIER)


def _get_enemy_combat_units(bot) -> list:
    """Get filtered enemy combat units for production analysis.
    
    Excludes workers, structures, and expired ghost units (age >= MEMORY_EXPIRY_TIME).
    """
    cached = bot.mediator.get_cached_enemy_army or []
    return [
        u for u in cached
        if u.type_id not in WORKER_TYPES
        and not u.is_structure
        and u.age < MEMORY_EXPIRY_TIME
    ]


def _compute_effectiveness(composition: dict, enemy_units: list) -> dict[UnitTypeId, float]:
    """Score each own unit type's effectiveness against the enemy army.
    
    For each unit type in our composition, sums COUNTER_TABLE bonuses against
    every observed enemy unit. Returns average bonus per enemy unit.
    
    Args:
        composition: Army composition dict {UnitTypeId: {"proportion": ..., "priority": ...}}
        enemy_units: Filtered enemy combat units
        
    Returns:
        {UnitTypeId: avg_effectiveness_score} for each type in composition
    """
    n_enemies = len(enemy_units)
    if n_enemies == 0:
        return {}
    
    scores: dict[UnitTypeId, float] = {}
    for own_type in composition:
        total = sum(
            COUNTER_TABLE.get((own_type, e.type_id), 0.0)
            for e in enemy_units
        )
        scores[own_type] = total / n_enemies
    return scores


def nudge_proportions(
    base_composition: dict,
    enemy_units: list,
) -> dict:
    """Apply counter-table-driven proportion nudges to a base army composition.
    
    Returns a new composition dict with adjusted proportions. The base composition
    is never mutated. If there's insufficient enemy data, returns base unchanged.
    
    Perf note: O(k * m) where k = unit types in comp (~5), m = enemy units.
    Runs once per macro cycle — negligible frame cost.
    
    Args:
        base_composition: The base army composition dict
        enemy_units: Filtered enemy combat units
        
    Returns:
        New composition dict with nudged proportions (sums to 1.0)
    """
    if len(enemy_units) < PRODUCTION_MIN_ENEMY_UNITS:
        return base_composition
    
    effectiveness = _compute_effectiveness(base_composition, enemy_units)
    if not effectiveness:
        return base_composition
    
    # Find the range of effectiveness scores to normalize nudges
    scores = list(effectiveness.values())
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score
    
    # All scores equal (or only one type) → no nudge needed
    if score_range < 0.1:
        return base_composition
    
    # Normalize each score to [-1, +1] range, then scale by MAX_NUDGE
    # Highest effectiveness → +MAX_NUDGE, lowest → -MAX_NUDGE
    mid = (max_score + min_score) / 2.0
    nudges: dict[UnitTypeId, float] = {}
    for unit_type, score in effectiveness.items():
        normalized = (score - mid) / (score_range / 2.0)  # -1 to +1
        nudges[unit_type] = normalized * PRODUCTION_MAX_NUDGE
    
    # Reorder priorities by effectiveness: most effective → priority 0 (built first)
    ranked_types = sorted(effectiveness, key=lambda t: effectiveness[t], reverse=True)
    priority_map: dict[UnitTypeId, int] = {ut: i for i, ut in enumerate(ranked_types)}
    
    # Apply nudges to base proportions and reordered priorities
    nudged: dict = {}
    for unit_type, info in base_composition.items():
        nudge = nudges.get(unit_type, 0.0)
        base_prop = info["proportion"]
        new_proportion = base_prop + nudge
        # Only apply min floor if unit has base proportion > 0 or got a positive nudge
        # Units at 0.0 base with no positive nudge should stay at 0.0
        if base_prop > 0 or nudge > 0:
            new_proportion = max(new_proportion, PRODUCTION_MIN_PROPORTION)
        else:
            new_proportion = 0.0
        nudged[unit_type] = {
            "proportion": new_proportion,
            "priority": priority_map.get(unit_type, info["priority"]),
        }
    
    # Re-normalize proportions to sum to exactly 1.0
    # Round each to 4 decimal places, then fix the last to absorb rounding error
    total = sum(v["proportion"] for v in nudged.values())
    if total > 0:
        items = list(nudged.values())
        for info in items:
            info["proportion"] = round(info["proportion"] / total, 4)
        # Force exact 1.0 sum (ARES SpawnController asserts isclose to 1.0)
        rounding_error = 1.0 - sum(v["proportion"] for v in items)
        items[-1]["proportion"] = round(items[-1]["proportion"] + rounding_error, 4)
    
    return nudged


def _get_gas_ratio(unit_type: UnitTypeId) -> float:
    """Get the gas fraction of a unit's total cost using ARES COST_DICT.
    
    Returns 0.0 for mineral-only units, 1.0 for gas-only units,
    ~0.5 for balanced cost units. Uses dynamic lookup — no hardcoded costs.
    
    Perf note: dict lookup, called once per unit type per macro cycle.
    """
    cost = COST_DICT.get(unit_type)
    if cost is None:
        return 0.5  # Unknown unit, assume balanced
    total = cost.minerals + cost.vespene
    if total == 0:
        return 0.0
    return cost.vespene / total


def reorder_priorities_by_resources(composition: dict, bot) -> dict:
    """Layer 1: Reorder unit priorities based on current resource balance.
    
    When mineral-rich/gas-poor: low-gas units (Zealots) get priority 0.
    When gas-rich/mineral-poor: high-gas units (HT, Disruptor) get priority 0.
    When balanced: return composition unchanged.
    
    This directly addresses the SpawnController break-on-unaffordable problem:
    by putting affordable units first in priority order, the controller builds
    them before hitting the unaffordable gas-heavy unit and breaking.
    
    The composition dict is never mutated — returns a new dict.
    
    Perf note: O(k log k) sort where k = unit types in comp (~5). Negligible.
    
    Args:
        composition: Army composition dict {UnitTypeId: {"proportion": ..., "priority": ...}}
        bot: Bot instance for resource access
        
    Returns:
        New composition dict with reordered priorities (proportions unchanged)
    """
    minerals = bot.minerals
    vespene = bot.vespene
    
    # Detect imbalance direction
    mineral_rich = minerals > RESOURCE_IMBALANCE_RATIO * max(vespene, 1)
    gas_rich = vespene > RESOURCE_IMBALANCE_RATIO * max(minerals, 1)
    
    if not mineral_rich and not gas_rich:
        return composition  # Balanced — keep original priorities
    
    # Sort unit types by gas_ratio: ascending if mineral-rich (cheap-gas first),
    # descending if gas-rich (expensive-gas first)
    unit_types = list(composition.keys())
    unit_types.sort(
        key=lambda ut: _get_gas_ratio(ut),
        reverse=gas_rich,
    )
    
    # Assign new priorities: 0 = highest (first in sorted order)
    reordered: dict = {}
    for new_priority, unit_type in enumerate(unit_types):
        info = composition[unit_type]
        reordered[unit_type] = {
            "proportion": info["proportion"],
            "priority": new_priority,
        }
    
    return reordered


def resource_pressure_nudge(composition: dict, bot) -> dict:
    """Layer 2: Shift proportions toward units whose cost matches available resources.
    
    When gas-starved (minerals >> gas): boost low-gas-ratio units, reduce high-gas ones.
    When mineral-starved (gas >> minerals): boost high-gas-ratio units, reduce low-gas ones.
    
    Chained after counter-table nudge_proportions() — both nudges compose additively.
    Max shift ±RESOURCE_PRESSURE_MAX_NUDGE (15%), clamped to PRODUCTION_MIN_PROPORTION floor,
    re-normalized to 1.0.
    
    Uses COST_DICT for dynamic gas-ratio lookup — no hardcoded costs.
    
    Perf note: O(k) where k = unit types in comp (~5). Negligible.
    
    Args:
        composition: Army composition dict (already counter-nudged)
        bot: Bot instance for resource access
        
    Returns:
        New composition dict with resource-pressure-adjusted proportions
    """
    minerals = bot.minerals
    vespene = bot.vespene
    
    # Detect imbalance
    gas_starved = minerals > RESOURCE_IMBALANCE_RATIO * max(vespene, 1)
    mineral_starved = vespene > RESOURCE_IMBALANCE_RATIO * max(minerals, 1)
    
    if not gas_starved and not mineral_starved:
        return composition  # Balanced — no pressure nudge needed
    
    # Compute gas ratios for all unit types in composition
    gas_ratios: dict[UnitTypeId, float] = {}
    for unit_type in composition:
        gas_ratios[unit_type] = _get_gas_ratio(unit_type)
    
    if not gas_ratios:
        return composition
    
    # Normalize gas ratios to [-1, +1] range for nudge direction
    ratios = list(gas_ratios.values())
    max_r, min_r = max(ratios), min(ratios)
    ratio_range = max_r - min_r
    
    if ratio_range < 0.05:
        return composition  # All units cost roughly the same — no nudge
    
    mid_r = (max_r + min_r) / 2.0
    
    # Build nudge per unit type
    nudged: dict = {}
    for unit_type, info in composition.items():
        gas_r = gas_ratios[unit_type]
        normalized = (gas_r - mid_r) / (ratio_range / 2.0)  # -1 to +1
        
        if gas_starved:
            # Boost low-gas units (negative normalized), penalize high-gas
            nudge = -normalized * RESOURCE_PRESSURE_MAX_NUDGE
        else:
            # mineral_starved: Boost high-gas units, penalize low-gas
            nudge = normalized * RESOURCE_PRESSURE_MAX_NUDGE
        
        new_proportion = info["proportion"] + nudge
        new_proportion = max(new_proportion, PRODUCTION_MIN_PROPORTION)
        nudged[unit_type] = {
            "proportion": new_proportion,
            "priority": info["priority"],
        }
    
    # Re-normalize to 1.0
    total = sum(v["proportion"] for v in nudged.values())
    if total > 0:
        items = list(nudged.values())
        for info in items:
            info["proportion"] = round(info["proportion"] / total, 4)
        rounding_error = 1.0 - sum(v["proportion"] for v in items)
        items[-1]["proportion"] = round(items[-1]["proportion"] + rounding_error, 4)
    
    return nudged


# No static upgrade list - use get_desired_upgrades() instead

def calculate_optimal_worker_count(bot) -> int:
    """
    Calculates the optimal number of workers based on available resources.
    Checks mineral contents and vespene contents to determine if mining positions are viable.
    
    Returns:
        int: The optimal number of workers to build
    """
    mineral_spots = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        for mineral in nearby_minerals:
            if mineral.mineral_contents > 100:  
                mineral_spots += 2
    
    gas_spots = 0
    for gas_building in bot.gas_buildings:
        if gas_building.vespene_contents > 100:
            gas_spots += 3
    
    # Add a small buffer for worker replacements
    worker_count = mineral_spots + gas_spots + 4
    
    return worker_count


def is_base_depleted(bot) -> bool:
    """Accurate base depletion using ARES data + python-sc2 mineral contents"""
    
    # Method 1: Check mineral contents directly (python-sc2)
    depleted_bases = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        if nearby_minerals:
            depleted_patches = len([m for m in nearby_minerals if m.mineral_contents < 300])
            if depleted_patches / len(nearby_minerals) > 0.5:
                depleted_bases += 1
    
    # Method 2: Use ARES worker assignment data
    mineral_assignments = bot.mediator.get_mineral_patch_to_list_of_workers
    available_patches = bot.mediator.get_num_available_mineral_patches
    
    # Count oversaturated patches (3+ workers per patch)
    oversaturated_count = sum(1 for workers in mineral_assignments.values() 
                             if len(workers) >= 3)
    
    gas_workers = len(bot.mediator.get_worker_to_vespene_dict)
    mining_workers = bot.workers.amount - gas_workers
    
    return (
        depleted_bases > 0                          # Some bases have depleted patches
        or available_patches < 4                    # Very few available patches  
        or (mining_workers > 0 and available_patches == 0)  # Workers but no patches
        or oversaturated_count > 2                  # Too many oversaturated patches
    )


def expansion_checker(bot, main_army) -> int:
    """
    Evaluates multiple factors to determine when to expand:
    1. Resource starvation (production idle due to lack of income)
    2. Base depletion (worker saturation with declining income)
    3. Army safety for expansion
    4. Spending efficiency using python-sc2 score metrics
    
    Returns the recommended expansion count.
    """
    current_bases = len(bot.townhalls)
    current_workers = bot.workers.amount
    optimal_workers = calculate_optimal_worker_count(bot)
    worker_saturation = current_workers / optimal_workers if optimal_workers > 0 else 0
    
    mineral_collection_rate = bot.state.score.collection_rate_minerals
    idle_production_time = bot.state.score.idle_production_time
    expansion_count = current_bases
    
    # Calculate spending efficiency (SQ-style)
    current_unspent = bot.minerals
    spending_efficiency = mineral_collection_rate / (current_unspent + 1) if mineral_collection_rate > 0 else 0
    
    # Safety check - only expand if safe (filter workers from both armies)
    own_combat_units = [u for u in bot.own_army if u.type_id not in WORKER_TYPES]
    enemy_combat_units = [u for u in bot.enemy_army if u.type_id not in WORKER_TYPES]
    army_safe = bot.mediator.can_win_fight(
        own_units=own_combat_units,
        enemy_units=enemy_combat_units,
        timing_adjust=True,
        good_positioning=False,
        workers_do_no_damage=True,
    ) in LOSS_MARGINAL_OR_BETTER
    
    if not army_safe:
        return expansion_count
    
    resource_starved = (
        idle_production_time > 30.0  # Production buildings idle for 30+ seconds
        and current_unspent < 500    # Low mineral bank
        and mineral_collection_rate > 0  # But we do have some income
    )
    
    bases_depleting = is_base_depleted(bot)
    
    # Spending efficiency thresholds (dynamic by game state)
    if bot.game_state == 0:      # Early: should spend quickly
        efficiency_threshold = 1.5
    elif bot.game_state == 1:    # Mid: more complex economy
        efficiency_threshold = 1.0  
    else:                        # Late: can bank for big investments
        efficiency_threshold = 0.5
    
    # Low spending efficiency = banking too much relative to income
    inefficient_spending = spending_efficiency < efficiency_threshold
    
    if resource_starved:
        expansion_count = current_bases + 1
    elif bases_depleting:
        expansion_count = current_bases + 1
    elif inefficient_spending and worker_saturation > 0.7:
        expansion_count = current_bases + 1
    
    # Game state-based fallback expansions (minimal safety net)
    if bot.game_state >= 1 and current_bases < 2:
        expansion_count = max(expansion_count, 2)
    elif bot.game_state >= 2 and current_bases < 3:
        expansion_count = max(expansion_count, 3)
    
    # Limit early expansion with small army
    if current_bases == 1 and len(main_army) < 5 and bot.game_state == 0:
        expansion_count = 1
    
    return expansion_count


def get_desired_upgrades(bot) -> list[UpgradeId]:
    """
    Returns dynamic upgrade list based on game state and structures.
    UpgradeController will automatically build required tech structures.
    """
    upgrades: list[UpgradeId] = []
    
    if not cy_upgrade_pending(bot, UpgradeId.WARPGATERESEARCH):
        upgrades.append(UpgradeId.WARPGATERESEARCH)
    
    if (bot.structures(UnitTypeId.ROBOTICSBAY) 
        and (bot.units(UnitTypeId.COLOSSUS) or cy_unit_pending(bot, UnitTypeId.COLOSSUS))):
        upgrades.append(UpgradeId.EXTENDEDTHERMALLANCE)
    
    if bot.structures(UnitTypeId.TWILIGHTCOUNCIL) or (len(bot.townhalls) >= 3 and bot.supply_used >= 54):
        upgrades.append(UpgradeId.CHARGE)
    
    # Blink: PvP gets it as soon as Twilight Council is up (Stalker-heavy comp).
    # PvT/PvZ get it after Thermal Lance is researched (Colossus range first).
    has_thermal_lance = cy_upgrade_pending(bot, UpgradeId.EXTENDEDTHERMALLANCE)
    if bot.enemy_race == Race.Protoss:
        if bot.structures(UnitTypeId.TWILIGHTCOUNCIL):
            upgrades.append(UpgradeId.BLINKTECH)
    elif has_thermal_lance:
        upgrades.append(UpgradeId.BLINKTECH)
    
    # Psi Storm: reactive trigger based on matchup.
    # PvT/PvZ: 2+ Vikings or Corruptors = anti-Colossus commitment, pivot to HT splash.
    # PvP: Only on Carriers — Phoenix/Tempest reduce Colossus via nudging but don't
    #       justify Storm. Carriers are the PvP signal that Storm splash is needed.
    if bot.structures(UnitTypeId.TEMPLARARCHIVE):
        if bot.enemy_race in {Race.Terran, Race.Zerg} and _count_enemy_anti_air(bot) >= 2:
            upgrades.append(UpgradeId.PSISTORMTECH)
        elif bot.enemy_race == Race.Protoss and _count_enemy_carriers(bot) >= 1:
            upgrades.append(UpgradeId.PSISTORMTECH)
    
    # Gate early upgrades if economy not ready (use centralized economy state)
    economy_state = get_economy_state(bot)
    if economy_state in ("recovery", "reduced"):
        return upgrades
    
    # Also gate if army is too small (need units before upgrades)
    if bot.supply_army < 15:
        return upgrades
    
    if bot.structures(UnitTypeId.FORGE) and ((len(bot.townhalls) >= 3 and bot.supply_used >= 56)):
        upgrades.extend([
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
            UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
            UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
            UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
        ])
    
    return upgrades


def require_warp_prism(bot) -> bool:
    """Returns True if WarpPrism should be built. Limits to 1 total."""
    # Check both forms: transport mode and phasing mode
    total_prisms = (bot.units(UnitTypeId.WARPPRISM).amount + 
                    bot.units(UnitTypeId.WARPPRISMPHASING).amount +
                    cy_unit_pending(bot, UnitTypeId.WARPPRISM))
    return (bot.structures(UnitTypeId.ROBOTICSFACILITY).ready
            and bot.structures(UnitTypeId.TEMPLARARCHIVE).ready
            and bot.structures(UnitTypeId.ROBOTICSBAY).ready
            and bot.supply_used >= 60
            and total_prisms == 0)


def require_shield_battery(bot) -> bool:
    """
    Returns True if shield battery should be built after cheese response.
    Limits to one battery total. Tech requirements checked by BuildStructure.
    """
    total_batteries = (bot.structures(UnitTypeId.SHIELDBATTERY).amount + 
                      cy_structure_pending_ares(bot, UnitTypeId.SHIELDBATTERY))
    
    return bot._used_cheese_response and total_batteries == 0


def get_shield_battery_base_location(bot) -> Point2:
    """
    Determines which base location to build shield battery at:
    - At natural if buildings or townhall exist there
    - At main base otherwise (uses static_defence placement)
    """
    natural_location = bot.mediator.get_own_nat
    
    natural_has_base = bot.townhalls.closer_than(10, natural_location).amount > 0
    natural_has_structures = bot.structures.closer_than(10, natural_location).amount > 0
    
    if natural_has_base or natural_has_structures:
        return natural_location
    else:
        # Use main base - BuildStructure with static_defence=True will find defensive spot
        return bot.start_location


def get_desired_gateway_count(bot) -> int:
    """
    Returns desired gateway/warpgate count based on supply and bases.
    Build order provides 3, scales to 5, then 8 for mid-game production.
    """
    if bot.supply_used >= 62 and len(bot.townhalls) >= 5:
        return 8
    elif bot.supply_used >= 54 and len(bot.townhalls) >= 3:
        return 5
    else:
        return 3


def get_desired_forge_count(bot) -> int:
    """
    Returns desired forge count based on bases and game state.
    Build 1 after first expansion, second in late game.
    """
    if len(bot.townhalls.ready) >= 4:
        return 2
    elif len(bot.townhalls.ready) >= 2:
        return 1
    else:
        return 0


def select_army_composition(bot, main_army: Units) -> dict:
    """
    Determines which army composition to use based on the current army state,
    enemy race, and observed enemy composition.
    
    Steps:
        1. Pick base composition from race + archon percentage (existing logic)
        2. Apply counter-table-driven proportion nudges based on enemy comp
        3. Cache the nudged result on bot._last_nudged_comp for debug display
    
    Returns:
        dict: The selected (and possibly nudged) army composition dictionary
    """
    if bot.enemy_race == Race.Protoss:
        selected_composition = PVP_ARMY_0
        army_1 = PVP_ARMY_1
        threshold = 0.30
    else:
        selected_composition = STANDARD_ARMY_0
        army_1 = STANDARD_ARMY_1
        threshold = 0.15
    
    # Switch composition when Archon percentage exceeds threshold
    if main_army and len(main_army) > 0:
        archon_count = sum(1 for unit in main_army if unit.type_id == UnitTypeId.ARCHON)
        archon_percentage = archon_count / len(main_army) if len(main_army) > 0 else 0
        
        if archon_percentage >= threshold:
            selected_composition = army_1
    
    # === Nudge pipeline: counter-table → resource-pressure → priority reorder ===
    
    # Step 1: Counter-table nudge (only if intel is fresh enough)
    intel = get_enemy_intel_quality(bot)
    if intel["has_intel"] and intel["freshness"] > STALE_INTEL_THRESHOLD:
        enemy_units = _get_enemy_combat_units(bot)
        comp = nudge_proportions(selected_composition, enemy_units)
    else:
        comp = selected_composition
    
    # Step 2: Resource-pressure nudge (shifts proportions toward affordable units)
    comp = resource_pressure_nudge(comp, bot)
    
    # Step 3: Priority reorder (puts affordable unit types first in SpawnController loop)
    comp = reorder_priorities_by_resources(comp, bot)
    
    # Cache for debug overlay
    bot._last_base_comp = selected_composition
    bot._last_nudged_comp = comp
    # Store resource pressure state for debug display
    minerals, vespene = bot.minerals, bot.vespene
    if minerals > RESOURCE_IMBALANCE_RATIO * max(vespene, 1):
        bot._resource_pressure = "GAS_STARVED"
    elif vespene > RESOURCE_IMBALANCE_RATIO * max(minerals, 1):
        bot._resource_pressure = "MIN_STARVED"
    else:
        bot._resource_pressure = "BALANCED"
    
    return comp


def _train_observers(bot) -> None:
    """Train observers to target count based on game state and matchup."""
    observer_count = (
        bot.units(UnitTypeId.OBSERVER).amount
        + bot.units(UnitTypeId.OBSERVERSIEGEMODE).amount
    )
    if cy_unit_pending(bot, UnitTypeId.OBSERVER) or not bot.can_afford(UnitTypeId.OBSERVER):
        return

    robotics_facilities = bot.structures(UnitTypeId.ROBOTICSFACILITY).ready
    if observer_count < 1:
        for facility in robotics_facilities:
            facility.train(UnitTypeId.OBSERVER)
            return

    if bot.game_state == 0:
        return

    target_count = 3 if bot.enemy_race in {Race.Zerg, Race.Terran} else 2
    if observer_count < target_count:
        for facility in robotics_facilities.idle:
            facility.train(UnitTypeId.OBSERVER)
            return


async def handle_macro(
    bot,
    main_army: Units,
    warp_prism: Units,
    freeflow: bool,
) -> None:
    """
    Main macro logic: builds units, keeps supply up, reacts to cheese, 
    or transitions to late game. Call this in your bot's on_step.
    """
    spawn_location = bot.natural_expansion
    production_location = bot.start_location
    
    worker_limit = 90 if bot.game_state >= 1 else 66
    optimal_worker_count = min(calculate_optimal_worker_count(bot), worker_limit)
    
    economy_state = get_economy_state(bot)
    
    # One-way transition from cheese defense to standard army
    if bot._used_cheese_response and not bot._transitioned_from_cheese:
        # Check transition conditions (one-way, never reverts)
        transition_conditions = (
            bot.game_state >= 1  # Mid-game
            or (not bot._under_attack and economy_state in ("moderate", "full"))  # Safe + healthy economy
        )
        if transition_conditions:
            bot._transitioned_from_cheese = True
    
    # Select army composition based on transition state
    if not bot._used_cheese_response or bot._transitioned_from_cheese:
        army_composition = select_army_composition(bot, main_army)
        expansion_count = expansion_checker(bot, main_army)
    else:
        army_composition = CHEESE_DEFENSE_ARMY
        expansion_count = 3
    
    if require_shield_battery(bot):
        base_location = get_shield_battery_base_location(bot)
        bot.register_behavior(
            BuildStructure(
                base_location=base_location,
                structure_id=UnitTypeId.SHIELDBATTERY,
                static_defence=True
            )
        )
    
    if require_warp_prism(bot):
        for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
            if bot.can_afford(UnitTypeId.WARPPRISM):
                facility.train(UnitTypeId.WARPPRISM)
                break
    
    # Scale gateways to desired count (3→5→8)
    # Allow in reduced+ economy so production capacity ramps before the moderate transition,
    # preventing a burst of 2+ gateways when economy recovers
    if economy_state in ("reduced", "moderate", "full"):
        desired_gates = get_desired_gateway_count(bot)
        current_gates = (bot.structures(UnitTypeId.GATEWAY).amount + 
                         bot.structures(UnitTypeId.WARPGATE).amount + 
                         cy_structure_pending_ares(bot, UnitTypeId.GATEWAY))
        
        if current_gates < desired_gates and bot.can_afford(UnitTypeId.GATEWAY):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.GATEWAY))
    
    # Scale forges to desired count (0→1→2) - moderate+ only (lower priority than gates)
    if economy_state in ("moderate", "full"):
        desired_forges = get_desired_forge_count(bot)
        current_forges = (bot.structures(UnitTypeId.FORGE).amount + 
                         cy_structure_pending_ares(bot, UnitTypeId.FORGE))
        
        if current_forges < desired_forges and bot.can_afford(UnitTypeId.FORGE):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.FORGE))
    
    macro_plan: MacroPlan = MacroPlan()
    
    # Always: workers and supply (all economy states)
    macro_plan.add(BuildWorkers(to_count=optimal_worker_count))
    macro_plan.add(AutoSupply(base_location=production_location))
    
    if economy_state == "recovery":
        pass
    else:
        # Reduced+: gas buildings and spawn from existing production
        macro_plan.add(GasBuildingController(to_count=len(bot.townhalls)*2, max_pending=2))
        
        spawn_target = warp_prism[0].position if warp_prism else spawn_location
        spawn_freeflow = True if bot._used_cheese_response else freeflow
        
        # During reduced economy, use freeflow mode so cheap affordable units
        # still get built (Layer 1 priority reorder puts them first).
        # Only fully skip spawning if we genuinely need to save for an expansion
        # that isn't already building.
        expansion_pending = cy_structure_pending_ares(bot, UnitTypeId.NEXUS) > 0
        if economy_state == "reduced" and not bot._under_attack and bot.minerals < 350 and not expansion_pending:
            # Genuinely tight and need to save for expansion — still produce in freeflow
            # so mineral-only units (Zealots) keep flowing via priority reordering
            macro_plan.add(SpawnController(army_composition, spawn_target=spawn_target, freeflow_mode=True))
        else:
            macro_plan.add(SpawnController(army_composition, spawn_target=spawn_target, freeflow_mode=spawn_freeflow))
        
        # Expansion logic: moderate+ gets full expansion, reduced gets safety net to 2 bases
        # Skip expansions when under attack - focus resources on defense
        if not bot._under_attack:
            if economy_state in ("moderate", "full"):
                macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1))
            elif len(bot.townhalls) < 2:
                # Safety net: always allow natural expansion even in reduced economy
                macro_plan.add(ExpansionController(to_count=2, max_pending=1))
        
        if economy_state in ("moderate", "full"):
            # Moderate+: upgrades
            macro_plan.add(UpgradeController(get_desired_upgrades(bot), base_location=production_location))
            
            if economy_state == "full":
                # Full: add new production buildings
                macro_plan.add(ProductionController(army_composition, base_location=production_location, add_production_at_bank=(450,450), ignore_below_proportion=0.3))
    
    bot.register_behavior(macro_plan)
    _train_observers(bot)
