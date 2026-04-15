"""Unit-level micro management functions.

Purpose: Reusable micro control logic for individual units in combat
Key Decisions: Pure functions that create and return CombatManeuver instances
Limitations: Assumes individual behavior pattern (one CombatManeuver per unit)
"""

from typing import List, Optional
import numpy as np

from sc2.unit import Unit
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.data import Race

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, StutterUnitBack, AMove, ShootTargetInRange, PathUnitToTarget,
    UseAOEAbility,
)
from cython_extensions import cy_closest_to, cy_in_attack_range, cy_distance_to, cy_find_units_center_mass
from bot.combat.target_scoring import select_target
from bot.constants import (
    DISRUPTOR_SQUAD_FOLLOW_DISTANCE, DISRUPTOR_SQUAD_TARGET_DISTANCE,
    HT_SQUAD_FOLLOW_DISTANCE, HT_SQUAD_TARGET_DISTANCE,
    HT_STORM_ENERGY_COST, HT_STORM_MIN_TARGETS,
    HT_FEEDBACK_ENERGY_COST, HT_FEEDBACK_RANGE, HT_FEEDBACK_MIN_ENEMY_ENERGY,
    FEEDBACK_TARGET_TYPES, HT_MERGE_ENERGY_THRESHOLD,
    HT_MERGE_COUNT_THRESHOLD, HT_MERGE_COUNT_THRESHOLD_PVP,
)


def micro_ranged_unit(
    unit: Unit,
    enemies: List[Unit],
    grid: np.ndarray,
    avoid_grid: np.ndarray,
    aggressive: bool = True
) -> CombatManeuver:
    """
    Create micro behaviors for a ranged unit.
    
    Uses weighted scoring (select_target) for target selection.
    Priority targeting is handled by score weights (UNIT_TYPE_VALUES),
    not a separate priority list.
    
    Args:
        unit: The ranged unit to control
        enemies: All enemy units in range (pre-filtered to attackable/reachable)
        grid: Ground grid for pathfinding
        avoid_grid: Avoidance grid for safety checks
        aggressive: Whether to fight aggressively or defensively
        
    Returns:
        CombatManeuver with appropriate behaviors added
    """
    maneuver = CombatManeuver()
    closest_enemy = cy_closest_to(unit.position, enemies)  # For kiting direction
    
    # Weighted scoring picks the best target considering distance, HP, type value, counter matchup
    best_target = select_target(unit, enemies)
    
    # ALWAYS add KeepUnitSafe FIRST with avoidance grid
    # This ensures units dodge dangerous abilities (disruptor shots, banelings, etc.)
    maneuver.add(KeepUnitSafe(unit, avoid_grid))
    
    # Defensive mode: prioritize kiting
    if not aggressive:
        maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=grid))
        return maneuver
    
    # Aggressive mode: stutter-step micro with scored targeting
    if not unit.weapon_ready:
        # Weapon on cooldown - kite back from closest threat
        maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=grid))
    else:
        # Weapon ready - shoot best scored target if in range, else advance
        if in_attack_range := cy_in_attack_range(unit, enemies):
            # Re-score only in-range enemies for the actual shot
            shoot_target = select_target(unit, in_attack_range)
            maneuver.add(ShootTargetInRange(unit=unit, targets=[shoot_target]))
        else:
            # Nothing in range - move towards best scored target
            maneuver.add(AMove(unit=unit, target=best_target.position))
    
    return maneuver


def micro_melee_unit(
    unit: Unit,
    enemies: List[Unit],
    avoid_grid: np.ndarray,
    fallback_position: Optional[Point2] = None,
    aggressive: bool = True
) -> CombatManeuver:
    """
    Create micro behaviors for a melee unit.
    
    Uses weighted scoring (select_target) for target selection.
    Priority targeting is handled by score weights, not a separate list.
    
    Args:
        unit: The melee unit to control
        enemies: All enemy units in range (pre-filtered to attackable/reachable)
        avoid_grid: Avoidance grid for safety (dodge disruptor shots, etc.)
        fallback_position: Position to move to if no targets
        aggressive: Whether to advance or hold position
        
    Returns:
        CombatManeuver with appropriate behaviors added
    """
    maneuver = CombatManeuver()
    
    # ALWAYS add KeepUnitSafe FIRST with avoidance grid
    # This ensures melee units dodge dangerous abilities (disruptor shots, banelings, etc.)
    maneuver.add(KeepUnitSafe(unit, avoid_grid))
    
    # Attack best scored target in range (both aggressive and defensive)
    if in_attack_range := cy_in_attack_range(unit, enemies):
        shoot_target = select_target(unit, in_attack_range)
        maneuver.add(ShootTargetInRange(unit=unit, targets=[shoot_target]))
    elif aggressive:
        # Aggressive mode: advance toward best scored target
        best_target = select_target(unit, enemies)
        maneuver.add(AMove(unit=unit, target=best_target.position))
    elif fallback_position:
        maneuver.add(AMove(unit=unit, target=fallback_position))
    # Defensive mode with nothing in range: do nothing (KeepUnitSafe handles retreat)
    
    return maneuver


def micro_disruptor(
    disruptor: Unit,
    enemies: List[Unit],
    friendly_units: List[Unit],
    avoid_grid: np.ndarray,
    grid: np.ndarray,
    bot,
    nova_manager,
    squad_position: Point2,
    ranged_center: Optional[Point2] = None,
) -> bool:
    """Handle disruptor nova firing and movement to stay with ground army.

    Disruptors are special: they can't attack, only fire novas.
    - Nova ready + enemies: let nova system take FULL control (no interference)
    - Nova on cooldown: stay safe in backlines, follow ranged line

    Uses two KeepUnitSafe layers like ranged/melee units:
      1. avoid_grid — dodge immediate danger (biles, disruptor shots, etc.)
      2. grid (ground influence) — kite away from enemy influence zones

    During combat, follows ranged_center (center of ranged units) so the
    disruptor mirrors the ranged line's movement. Falls back to squad_position
    when no ranged center is available.

    Args:
        disruptor: The Disruptor unit to control
        enemies: Enemy units (filtered for disruptor targeting)
        friendly_units: Friendly units (to avoid friendly fire)
        avoid_grid: Avoidance grid for immediate danger
        grid: Ground grid with enemy influence (for kiting away from enemy zones)
        bot: Bot instance (for use_disruptor_nova access)
        nova_manager: NovaManager instance for coordination
        squad_position: Center of the full squad (fallback when no ranged_center)
        ranged_center: Center of ranged units during combat (preferred follow target)

    Returns:
        True if nova fired or behavior registered, False otherwise
    """

    # Check if nova is ready AND enemies are present
    nova_ready = AbilityId.EFFECT_PURIFICATIONNOVA in disruptor.abilities

    if nova_ready and enemies:
        # Nova system takes FULL control - don't register any ARES maneuver
        # This avoids conflict between KeepUnitSafe and nova targeting
        try:
            result = bot.use_disruptor_nova.execute(
                disruptor, enemies, friendly_units, nova_manager
            )
            if result:  # Nova fired successfully
                return True
        except Exception as e:
            print(f"DEBUG ERROR in disruptor nova firing: {e}")

    # Nova on cooldown OR no enemies - disruptor should stay safe in backlines
    maneuver = CombatManeuver()

    # Layer 1: Dodge immediate danger (biles, disruptor shots, storms, etc.)
    maneuver.add(KeepUnitSafe(disruptor, avoid_grid))

    # Layer 2: Kite away from enemy influence zones (same pattern as ranged/melee units)
    maneuver.add(KeepUnitSafe(disruptor, grid))

    # Follow the ranged center during combat, squad center otherwise
    follow_target = ranged_center if ranged_center is not None else squad_position
    distance_to_follow = cy_distance_to(disruptor.position, follow_target)
    if distance_to_follow > DISRUPTOR_SQUAD_FOLLOW_DISTANCE:
        maneuver.add(PathUnitToTarget(
            unit=disruptor,
            grid=bot.mediator.get_ground_grid,
            target=follow_target,
            success_at_distance=DISRUPTOR_SQUAD_TARGET_DISTANCE,
        ))

    bot.register_behavior(maneuver)
    return True


def merge_high_templars(bot) -> None:
    """Merge HT pairs into Archons via two independent triggers (OR logic).

    Trigger 1 — Energy: Spent HTs (energy < threshold) merge regardless
    of total count. These HTs have done their job and should become Archons.

    Trigger 2 — Count: When we have more HTs than the count threshold,
    merge one closest pair (lowest energy first) to convert surplus into
    Archons. Only one pair per frame to avoid over-merging.

    Count threshold is per-matchup (lower in PvP where HTs are primarily
    Archon-in-waiting). The Archon-percentage switch to army_1 handles
    the loop brake, not the count gate.

    Pairs are selected by proximity: sort candidates by energy (lowest
    first), then find the closest neighbor. This produces Archons that
    arrive at the fight faster than arbitrary pairing.

    Uses request_archon_morph for a proper paired command so both HTs
    get a single merge order (no order conflicts).

    Call once per frame from the combat loop (not macro — HT lifecycle
    belongs with HT micro logic).
    """
    all_hts = bot.units(UnitTypeId.HIGHTEMPLAR).ready
    count_threshold = (
        HT_MERGE_COUNT_THRESHOLD_PVP
        if bot.enemy_race == Race.Protoss
        else HT_MERGE_COUNT_THRESHOLD
    )

    # Track which HTs are already being merged to avoid double-merging
    merging_tags: set[int] = set()

    # --- Trigger 1: Energy (spent HTs merge regardless of count) ---
    low_energy_hts = [
        ht for ht in all_hts
        if ht.energy < HT_MERGE_ENERGY_THRESHOLD
        and not (ht.orders and ht.orders[0].ability.id == AbilityId.MORPH_ARCHON)
    ]
    low_energy_hts.sort(key=lambda ht: ht.energy)
    while len(low_energy_hts) >= 2:
        ht_a = low_energy_hts.pop(0)
        closest_idx, closest_dist = 0, float("inf")
        for i, ht_b in enumerate(low_energy_hts):
            dist = cy_distance_to(ht_a.position, ht_b.position)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        ht_b = low_energy_hts.pop(closest_idx)
        bot.request_archon_morph([ht_a, ht_b])
        merging_tags.add(ht_a.tag)
        merging_tags.add(ht_b.tag)

    # --- Trigger 2: Count (over-stocked → merge one closest pair) ---
    if len(all_hts) > count_threshold:
        # Only consider HTs not already merging from trigger 1
        surplus_hts = [
            ht for ht in all_hts
            if ht.tag not in merging_tags
            and not (ht.orders and ht.orders[0].ability.id == AbilityId.MORPH_ARCHON)
        ]
        if len(surplus_hts) >= 2:
            surplus_hts.sort(key=lambda ht: ht.energy)
            ht_a = surplus_hts[0]
            closest_idx, closest_dist = 1, float("inf")
            for i, ht_b in enumerate(surplus_hts[1:], start=1):
                dist = cy_distance_to(ht_a.position, ht_b.position)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            ht_b = surplus_hts[closest_idx]
            bot.request_archon_morph([ht_a, ht_b])


def micro_high_templar(
    ht: Unit,
    enemies: List[Unit],
    avoid_grid: np.ndarray,
    grid: np.ndarray,
    bot,
    squad_position: Point2,
    ranged_center: Optional[Point2] = None,
) -> bool:
    """Handle High Templar abilities and safe army-following.

    Priority: Psi Storm > Feedback > stay safe + follow army.
    HTs are fragile casters that should stay behind the army.
    Skips HTs that are currently merging to archon (MORPH_ARCHON order).

    Uses two KeepUnitSafe layers like ranged/melee units:
      1. avoid_grid — dodge immediate danger (biles, disruptor shots, etc.)
      2. grid (ground influence) — kite away from enemy influence zones

    During combat, follows ranged_center (center of ranged units) so the
    HT mirrors the ranged line's movement. Falls back to squad_position
    when no ranged center is available.

    Args:
        ht: The High Templar unit
        enemies: Nearby enemy units (all visible)
        avoid_grid: Avoidance grid for immediate danger
        grid: Ground grid with enemy influence (for kiting away from enemy zones)
        bot: Bot instance
        squad_position: Center of the full squad (fallback when no ranged_center)
        ranged_center: Center of ranged units during combat (preferred follow target)

    Returns:
        True if behavior registered
    """
    # Skip if HT is currently merging to archon
    if ht.orders and ht.orders[0].ability.id == AbilityId.MORPH_ARCHON:
        return False

    # Priority 1: Psi Storm — ARES UseAOEAbility finds optimal cast position,
    # avoids friendly fire and duplicate storms automatically
    if ht.energy >= HT_STORM_ENERGY_COST and enemies:
        storm_behavior = UseAOEAbility(
            unit=ht,
            ability_id=AbilityId.PSISTORM_PSISTORM,
            targets=enemies,
            min_targets=HT_STORM_MIN_TARGETS,
            avoid_own_ground=True,
            avoid_own_flying=True,
        )
        if storm_behavior.execute(bot, bot.config, bot.mediator):
            return True

    # Priority 2: Feedback — only on high-value caster types
    if ht.energy >= HT_FEEDBACK_ENERGY_COST and enemies:
        feedback_targets = [
            e for e in enemies
            if e.type_id in FEEDBACK_TARGET_TYPES
            and e.energy >= HT_FEEDBACK_MIN_ENEMY_ENERGY
            and cy_distance_to(ht.position, e.position) <= HT_FEEDBACK_RANGE
        ]
        if feedback_targets:
            best_target = max(feedback_targets, key=lambda e: e.energy)
            ht(AbilityId.FEEDBACK_FEEDBACK, best_target)
            return True

    # No cast opportunity — stay safe and follow army
    maneuver = CombatManeuver()

    # Layer 1: Dodge immediate danger (biles, disruptor shots, storms, etc.)
    maneuver.add(KeepUnitSafe(ht, avoid_grid))

    # Layer 2: Kite away from enemy influence zones (same pattern as ranged/melee units)
    maneuver.add(KeepUnitSafe(ht, grid))

    # Follow the ranged center during combat, squad center otherwise
    follow_target = ranged_center if ranged_center is not None else squad_position
    distance_to_follow = cy_distance_to(ht.position, follow_target)
    if distance_to_follow > HT_SQUAD_FOLLOW_DISTANCE:
        maneuver.add(PathUnitToTarget(
            unit=ht,
            grid=bot.mediator.get_ground_grid,
            target=follow_target,
            success_at_distance=HT_SQUAD_TARGET_DISTANCE,
        ))

    bot.register_behavior(maneuver)
    return True


def micro_sentry(
    sentry: Unit,
    friendly_units: List[Unit],
    enemies: List[Unit],
    avoid_grid: np.ndarray,
    grid: np.ndarray,
    bot,
    squad_position: Point2,
    ranged_center: Optional[Point2] = None,
) -> bool:
    """Handle Sentry Guardian Shield and safe army-following.

    Sentries spread coverage across the army — avoid overlapping shields.
    - Priority: Guardian Shield (if ranged enemies present + space from other shields)
    - Otherwise: stay safe (two-layer: avoid grid + ground grid) and follow army

    Uses two KeepUnitSafe layers like ranged/melee units:
      1. avoid_grid — dodge immediate danger (biles, disruptor shots, etc.)
      2. grid (ground influence) — kite away from enemy influence zones

    During combat, follows ranged_center (center of ranged units) so the
    sentry mirrors the ranged line's movement instead of the full squad center
    (which includes melee units at the front). Falls back to squad_position
    when no ranged center is available (e.g. early game, no-enemy movement).

    Args:
        sentry: The Sentry unit
        friendly_units: Nearby friendly units to potentially cover
        enemies: Nearby enemy units (already filtered for combat-relevant types)
        avoid_grid: Avoidance grid for immediate danger (biles, disruptor, etc.)
        grid: Ground grid with enemy influence (for kiting away from enemy zones)
        bot: Bot instance
        squad_position: Center of the full squad (fallback when no ranged_center)
        ranged_center: Center of ranged units during combat (preferred follow target)

    Returns:
        True if behavior registered
    """
    from bot.constants import (
        MELEE_RANGE_THRESHOLD,
        SENTRY_SQUAD_FOLLOW_DISTANCE,
        SENTRY_SQUAD_TARGET_DISTANCE,
        GUARDIAN_SHIELD_ENERGY_COST,
        GUARDIAN_SHIELD_OVERLAP_DISTANCE,
    )
    from sc2.ids.buff_id import BuffId

    # Skip if already shielded (can't double-cast)
    has_shield = sentry.has_buff(BuffId.GUARDIANSHIELD)

    # Guardian Shield only reduces ranged damage — skip if no ranged enemies present
    has_ranged_enemies = any(
        u.ground_range > MELEE_RANGE_THRESHOLD for u in enemies
    )

    # Check if another friendly sentry nearby already has shield active
    shield_overlap = False
    if (
        sentry.energy >= GUARDIAN_SHIELD_ENERGY_COST
        and has_ranged_enemies
        and not has_shield
    ):
        for u in friendly_units:
            if u.type_id == UnitTypeId.SENTRY and u.tag != sentry.tag:
                if u.has_buff(BuffId.GUARDIANSHIELD):
                    dist = cy_distance_to(sentry.position, u.position)
                    if dist < GUARDIAN_SHIELD_OVERLAP_DISTANCE:
                        shield_overlap = True
                        break

        # Cast shield if no overlap and enemies present
        if not shield_overlap:
            sentry(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD)
            return True

    # No shield opportunity — stay safe and follow army
    maneuver = CombatManeuver()

    # Layer 1: Dodge immediate danger (biles, disruptor shots, storms, etc.)
    maneuver.add(KeepUnitSafe(sentry, avoid_grid))

    # Layer 2: Kite away from enemy influence zones (same pattern as ranged/melee units)
    maneuver.add(KeepUnitSafe(sentry, grid))

    # Follow the ranged center during combat, squad center otherwise
    follow_target = ranged_center if ranged_center is not None else squad_position
    distance_to_follow = cy_distance_to(sentry.position, follow_target)
    if distance_to_follow > SENTRY_SQUAD_FOLLOW_DISTANCE:
        maneuver.add(PathUnitToTarget(
            unit=sentry,
            grid=bot.mediator.get_ground_grid,
            target=follow_target,
            success_at_distance=SENTRY_SQUAD_TARGET_DISTANCE,
        ))

    bot.register_behavior(maneuver)
    return True
