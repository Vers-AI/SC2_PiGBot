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

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, StutterUnitBack, AMove, ShootTargetInRange
)

from bot.constants import PRIORITY_TARGET_TYPES

from cython_extensions import cy_closest_to, cy_in_attack_range, cy_distance_to, cy_find_units_center_mass


def micro_ranged_unit(
    unit: Unit,
    enemies: List[Unit],
    priority_targets: List[Unit],
    grid: np.ndarray,
    avoid_grid: np.ndarray,
    aggressive: bool = True
) -> CombatManeuver:
    """
    Create micro behaviors for a ranged unit.
    
    Implements priority targeting with stutter-step micro:
    - Aggressive: engage and advance when weapon ready
    - Defensive: stay safe and kite regardless of weapon state
    
    Args:
        unit: The ranged unit to control
        enemies: All enemy units in range
        priority_targets: High-priority targets (tanks, colossi, etc.)
        grid: Ground grid for pathfinding
        avoid_grid: Avoidance grid for safety checks
        aggressive: Whether to fight aggressively or defensively
        
    Returns:
        CombatManeuver with appropriate behaviors added
    """
    maneuver = CombatManeuver()
    closest_enemy = cy_closest_to(unit.position, enemies)
    
    # ALWAYS add KeepUnitSafe FIRST with avoidance grid
    # This ensures units dodge dangerous abilities (disruptor shots, banelings, etc.)
    maneuver.add(KeepUnitSafe(unit, avoid_grid))
    
    # Defensive mode: prioritize kiting
    if not aggressive:
        maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=grid))
        return maneuver
    
    # Aggressive mode: standard stutter-step micro
    if not unit.weapon_ready:
        # Weapon on cooldown - kite back
        maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=grid))
    else:
        # Weapon ready - engage with priority targeting
        if in_attack_range_priority := cy_in_attack_range(unit, priority_targets):
            maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_priority))
        elif in_attack_range_any := cy_in_attack_range(unit, enemies):
            maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_any))
        else:
            # Nothing in range - move towards closest enemy
            maneuver.add(AMove(unit=unit, target=closest_enemy.position))
    
    return maneuver


def micro_melee_unit(
    unit: Unit,
    enemies: List[Unit],
    priority_targets: List[Unit],
    avoid_grid: np.ndarray,
    fallback_position: Optional[Point2] = None,
    aggressive: bool = True
) -> CombatManeuver:
    """
    Create micro behaviors for a melee unit.
    
    Implements simple priority targeting:
    - Aggressive: advance and attack targets
    - Defensive: only attack targets already in range
    
    Args:
        unit: The melee unit to control
        enemies: All enemy units in range
        priority_targets: High-priority targets (tanks, colossi, etc.)
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
    
    # Attack targets in range (both aggressive and defensive)
    if in_attack_range_priority := cy_in_attack_range(unit, priority_targets):
        maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_priority))
    elif in_attack_range_any := cy_in_attack_range(unit, enemies):
        maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_any))
    elif aggressive:
        # Aggressive mode: advance toward targets
        if priority_targets:
            closest_priority = cy_closest_to(unit.position, priority_targets)
            maneuver.add(AMove(unit=unit, target=closest_priority.position))
        elif fallback_position:
            maneuver.add(AMove(unit=unit, target=fallback_position))
    # Defensive mode: don't advance, already handled attacks in range above
    
    return maneuver


def micro_disruptor(
    disruptor: Unit,
    enemies: List[Unit],
    friendly_units: List[Unit],
    avoid_grid: np.ndarray,
    bot,
    nova_manager,
    squad_position: Point2
) -> bool:
    """
    Handle disruptor nova firing and movement to stay with ground army.
    
    Disruptors are special: they can't attack, only fire novas.
    - Nova ready + enemies: let nova system take FULL control (no interference)
    - Nova on cooldown: stay safe in backlines, follow army from behind
    
    Args:
        disruptor: The Disruptor unit to control
        enemies: Enemy units (filtered for disruptor targeting)
        friendly_units: Friendly units (to avoid friendly fire)
        avoid_grid: Avoidance grid for safety
        bot: Bot instance (for use_disruptor_nova access)
        nova_manager: NovaManager instance for coordination
        squad_position: Center of the ground army to follow
        
    Returns:
        True if nova fired or behavior registered, False otherwise
    """
    from sc2.ids.ability_id import AbilityId
    from ares.behaviors.combat.individual import PathUnitToTarget
    from bot.constants import DISRUPTOR_SQUAD_FOLLOW_DISTANCE, DISRUPTOR_SQUAD_TARGET_DISTANCE
    
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
    
    # KeepUnitSafe FIRST - dodge dangerous abilities
    maneuver.add(KeepUnitSafe(disruptor, avoid_grid))
    
    # Follow army from behind (stay with ground units, not air)
    ground_units = [u for u in friendly_units if not u.is_flying]
    if ground_units:
        ground_center, _ = cy_find_units_center_mass(ground_units, 10.0)
        distance_to_center = cy_distance_to(disruptor.position, ground_center)
        
        # If too far from ground army, move towards center
        if distance_to_center > DISRUPTOR_SQUAD_FOLLOW_DISTANCE:
            maneuver.add(PathUnitToTarget(
                unit=disruptor,
                grid=bot.mediator.get_ground_grid,
                target=Point2(ground_center),
                success_at_distance=DISRUPTOR_SQUAD_TARGET_DISTANCE
            ))
    else:
        # Fallback: stay with squad position if no ground units
        distance_to_squad = cy_distance_to(disruptor.position, squad_position)
        if distance_to_squad > DISRUPTOR_SQUAD_FOLLOW_DISTANCE:
            maneuver.add(PathUnitToTarget(
                unit=disruptor,
                grid=bot.mediator.get_ground_grid,
                target=squad_position,
                success_at_distance=DISRUPTOR_SQUAD_TARGET_DISTANCE
            ))
    
    bot.register_behavior(maneuver)
    return True


def get_priority_targets(enemies: List[Unit]) -> List[Unit]:
    """
    Filter enemies to get high-priority targets.
    
    Args:
        enemies: All enemy units
        
    Returns:
        List of high-priority targets (tanks, colossi, etc.)
    """
    return [u for u in enemies if u.type_id in PRIORITY_TARGET_TYPES]


