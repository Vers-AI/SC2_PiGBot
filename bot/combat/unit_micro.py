"""Unit-level micro management functions.

Purpose: Reusable micro control logic for individual units in combat
Key Decisions: Pure functions that create and return CombatManeuver instances;
    Stalker blink uses concave-aware positioning when available
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
    UseAOEAbility, UseAbility,
)
from cython_extensions import cy_closest_to, cy_in_attack_range, cy_distance_to, cy_find_units_center_mass, cy_towards
from bot.combat.target_scoring import select_target
from bot.constants import (
    DISRUPTOR_SQUAD_FOLLOW_DISTANCE, DISRUPTOR_SQUAD_TARGET_DISTANCE,
    HT_SQUAD_FOLLOW_DISTANCE, HT_SQUAD_TARGET_DISTANCE,
    HT_STORM_ENERGY_COST, HT_STORM_MIN_TARGETS,
    HT_FEEDBACK_ENERGY_COST, HT_FEEDBACK_RANGE, HT_FEEDBACK_MIN_ENEMY_ENERGY,
    FEEDBACK_TARGET_TYPES, HT_MERGE_ENERGY_THRESHOLD,
    HT_MERGE_COUNT_THRESHOLD, HT_MERGE_COUNT_THRESHOLD_PVP,
    STALKER_BLINK_HEALTH_THRESHOLD, STALKER_BLINK_RANGE,
)


def micro_stalker(
    stalker: Unit,
    enemies: List[Unit],
    grid: np.ndarray,
    avoid_grid: np.ndarray,
    aggressive: bool = True,
    squad_center: Optional[Point2] = None,
    enemy_center: Optional[Point2] = None,
    ranged_units: Optional[List[Unit]] = None,
) -> CombatManeuver:
    """Stalker micro with blink-when-low behavior.

    When a Stalker's combined health+shields drop below the threshold and
    Blink is off cooldown, the Stalker blinks along the concave line
    (perpendicular to the approach direction) toward the far edge of the
    formation. This keeps the Stalker in the fight at a safer position
    instead of stutter-stepping back (which is slower and still takes hits).

    If no concave geometry is available (no squad/enemy center), falls back
    to blinking directly away from the closest enemy.

    When blink isn't needed (healthy or on cooldown), delegates to
    micro_ranged_unit for standard stutter-step behavior.

    Args:
        stalker: The Stalker unit to control
        enemies: All enemy units in range (pre-filtered to attackable/reachable)
        grid: Ground grid for pathfinding
        avoid_grid: Avoidance grid for safety checks
        aggressive: Whether to fight aggressively or defensively
        squad_center: Center of the squad (for concave direction)
        enemy_center: Center of nearby enemies (for concave direction)
        ranged_units: Other ranged units in the squad (for concave edge calculation)

    Returns:
        CombatManeuver with appropriate behaviors added
    """
    # Check if blink is available and stalker is low health
    # shield_health_percentage = (health + shield) / (health_max + shield_max), includes build progress
    health_ratio = stalker.shield_health_percentage
    blink_ready = AbilityId.EFFECT_BLINK_STALKER in stalker.abilities

    if blink_ready and health_ratio < STALKER_BLINK_HEALTH_THRESHOLD and enemies:
        # Compute blink direction: along the concave line toward the far edge
        blink_target = _compute_blink_target(
            stalker, enemies, squad_center, enemy_center, ranged_units
        )

        if blink_target is not None:
            maneuver = CombatManeuver()
            # Blink is the priority action — UseAbility checks cooldown internally
            # and returns True if executed, short-circuiting the rest of the maneuver.
            # If blink fails (e.g. target unpathable), falls through to KeepUnitSafe.
            maneuver.add(UseAbility(
                ability=AbilityId.EFFECT_BLINK_STALKER,
                unit=stalker,
                target=blink_target,
            ))
            # Safety fallback if blink doesn't execute
            maneuver.add(KeepUnitSafe(stalker, avoid_grid))
            return maneuver

    # No blink needed — fall through to standard ranged micro
    return micro_ranged_unit(
        unit=stalker,
        enemies=enemies,
        grid=grid,
        avoid_grid=avoid_grid,
        aggressive=aggressive,
    )


def _compute_blink_target(
    stalker: Unit,
    enemies: List[Unit],
    squad_center: Optional[Point2],
    enemy_center: Optional[Point2],
    ranged_units: Optional[List[Unit]],
) -> Optional[Point2]:
    """Compute where a low-health Stalker should blink to.

    Strategy: blink along the concave line (perpendicular to approach
    direction) toward the far edge of the formation. This keeps the
    Stalker contributing DPS from a safer position at the concave edge
    rather than retreating entirely.

    Falls back to blinking away from the closest enemy when concave
    geometry isn't available.

    Args:
        stalker: The Stalker unit that will blink
        enemies: Nearby enemy units
        squad_center: Center of the squad (for approach direction)
        enemy_center: Center of nearby enemies (for approach direction)
        ranged_units: Other ranged units in the squad (for edge calculation)

    Returns:
        Point2 blink target, or None if no valid target found
    """
    pos = stalker.position

    # Determine approach direction (squad → enemy) for concave line
    if squad_center is not None and enemy_center is not None:
        length = cy_distance_to(squad_center, enemy_center)

        if length > 0.01:
            # Approach direction unit vector
            dx = enemy_center.x - squad_center.x
            dy = enemy_center.y - squad_center.y
            approach_x = dx / length
            approach_y = dy / length

            # Perpendicular axis (left/right relative to approach)
            perp_x = -approach_y
            perp_y = approach_x

            # Find which side of the concave the stalker is on,
            # then blink toward the far edge of the formation
            if ranged_units and len(ranged_units) >= 2:
                # Compute lateral offsets for all ranged units
                offsets = []
                for u in ranged_units:
                    rel_x = u.position.x - squad_center.x
                    rel_y = u.position.y - squad_center.y
                    lateral = rel_x * perp_x + rel_y * perp_y
                    offsets.append(lateral)

                # The stalker's own lateral offset
                stalker_lateral = (pos.x - squad_center.x) * perp_x + (pos.y - squad_center.y) * perp_y
                # Stalker's depth along the approach axis (positive = toward enemy)
                stalker_depth = (pos.x - squad_center.x) * approach_x + (pos.y - squad_center.y) * approach_y

                # Blink toward the far edge of the formation from the stalker's position
                # If stalker is on the left side, blink further left; if right, further right
                min_offset = min(offsets)
                max_offset = max(offsets)

                if stalker_lateral >= 0:
                    # Stalker is on the right side — blink toward right edge
                    target_lateral = max_offset
                else:
                    # Stalker is on the left side — blink toward left edge
                    target_lateral = min_offset

                # Target point: on the concave edge, at the stalker's depth,
                # shifted slightly back (away from enemy) for safety
                retreat_tiles = 2.0
                target_x = squad_center.x + perp_x * target_lateral + approach_x * (stalker_depth - retreat_tiles)
                target_y = squad_center.y + perp_y * target_lateral + approach_y * (stalker_depth - retreat_tiles)
                blink_target = Point2((target_x, target_y))

                # Clamp to blink range from current position
                dist = cy_distance_to(pos, blink_target)
                if dist > STALKER_BLINK_RANGE:
                    blink_target = Point2(cy_towards(pos, blink_target, STALKER_BLINK_RANGE))

                return blink_target

    # Fallback: blink directly away from closest enemy
    closest_enemy = cy_closest_to(pos, enemies)
    if closest_enemy is not None:
        # Point away from enemy at blink range
        away_point = Point2((
            2 * pos.x - closest_enemy.position.x,
            2 * pos.y - closest_enemy.position.y,
        ))
        return Point2(cy_towards(pos, away_point, STALKER_BLINK_RANGE))

    return None


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
    if bot.enemy_race == Race.Protoss:
        # PvP: if enemy has Carriers, hold HTs for Storm instead of merging
        enemy_has_carriers = any(
            u.type_id == UnitTypeId.CARRIER
            for u in (bot.mediator.get_cached_enemy_army or [])
        )
        count_threshold = HT_MERGE_COUNT_THRESHOLD if enemy_has_carriers else HT_MERGE_COUNT_THRESHOLD_PVP
    else:
        count_threshold = HT_MERGE_COUNT_THRESHOLD

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
    ff_assignments: Optional[dict[int, list[Point2]]] = None,
    gs_approved: bool = False,
) -> bool:
    """Handle Sentry abilities and safe army-following.

    Priority order:
      1. Force Field split (if this sentry was assigned FF positions)
      2. Guardian Shield (if squad-level assignment approved this sentry)
      3. Stay safe (two-layer: avoid grid + ground grid) and follow army

    Force Field split is computed once per squad (in combat.py) across all
    sentries pooling their energy, then each sentry executes its assigned
    casts. This ensures coordinated FF placement that a single sentry
    couldn't achieve alone.

    Guardian Shield assignment is also computed once per squad (in
    combat.py) via _compute_guardian_shield_assignments(). Only the
    minimum sentries needed to cover the squad are approved to cast,
    preventing all sentries from casting simultaneously.

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
        ff_assignments: Dict mapping sentry tag → list of FF positions to cast.
            Pre-computed by compute_ff_split() in combat.py. None = no split.
        gs_approved: Whether this sentry was approved by the squad-level
            Guardian Shield assignment to cast. If False, skip casting.

    Returns:
        True if behavior registered
    """
    from bot.constants import (
        SENTRY_SQUAD_FOLLOW_DISTANCE,
        SENTRY_SQUAD_TARGET_DISTANCE,
    )
    from sc2.ids.buff_id import BuffId

    # Priority 1: Force Field split — cast assigned FFs if available
    if ff_assignments and sentry.tag in ff_assignments:
        positions = ff_assignments[sentry.tag]
        if positions:
            # Cast all assigned FFs for this sentry
            # (energy was already validated in compute_ff_split)
            # First cast is immediate (queue=False), subsequent casts are queued
            # so SC2 executes them in sequence instead of overwriting each other
            for i, pos in enumerate(positions):
                sentry(AbilityId.FORCEFIELD_FORCEFIELD, pos, queue=(i > 0))
            return True

    # Priority 2: Guardian Shield
    # Squad-level assignment (computed in combat.py) determines which
    # sentries cast. Only approved sentries spend energy, ensuring
    # minimum shields to cover the squad without over-casting.
    if gs_approved and not sentry.has_buff(BuffId.GUARDIANSHIELD):
        sentry(AbilityId.GUARDIANSHIELD_GUARDIANSHIELD)
        return True

    # No cast opportunity — stay safe and follow army
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
