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

from cython_extensions import cy_closest_to, cy_in_attack_range


def micro_ranged_unit(
    unit: Unit,
    enemies: List[Unit],
    priority_targets: List[Unit],
    grid: np.ndarray,
    avoid_grid: np.ndarray
) -> CombatManeuver:
    """
    Create micro behaviors for a ranged unit.
    
    Implements priority targeting with stutter-step micro:
    - Weapon on cooldown: safety check + kite back
    - Weapon ready: prioritize high-value targets
    
    Args:
        unit: The ranged unit to control
        enemies: All enemy units in range
        priority_targets: High-priority targets (tanks, colossi, etc.)
        grid: Ground grid for pathfinding
        avoid_grid: Avoidance grid for safety checks
        
    Returns:
        CombatManeuver with appropriate behaviors added
    """
    maneuver = CombatManeuver()
    closest_enemy = cy_closest_to(unit.position, enemies)
    
    if not unit.weapon_ready:
        # Weapon on cooldown - explicit safety check then kite back
        maneuver.add(KeepUnitSafe(unit, avoid_grid))
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
    fallback_position: Optional[Point2] = None
) -> CombatManeuver:
    """
    Create micro behaviors for a melee unit.
    
    Implements simple priority targeting:
    - Attacks priority targets first if in range
    - Falls back to any target in range
    - Moves toward priority target or general position
    
    Args:
        unit: The melee unit to control
        enemies: All enemy units in range
        priority_targets: High-priority targets (tanks, colossi, etc.)
        fallback_position: Position to move to if no targets
        
    Returns:
        CombatManeuver with appropriate behaviors added
    """
    maneuver = CombatManeuver()
    
    # Simple priority hierarchy for melee: Priority -> Any -> Move
    if in_attack_range_priority := cy_in_attack_range(unit, priority_targets):
        maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_priority))
    elif in_attack_range_any := cy_in_attack_range(unit, enemies):
        maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_any))
    else:
        # Move towards priority target or fallback position
        if priority_targets:
            closest_priority = cy_closest_to(unit.position, priority_targets)
            maneuver.add(AMove(unit=unit, target=closest_priority.position))
        elif fallback_position:
            maneuver.add(AMove(unit=unit, target=fallback_position))
    
    return maneuver


def micro_disruptor(
    disruptor: Unit,
    enemies: List[Unit],
    friendly_units: List[Unit],
    avoid_grid: np.ndarray,
    bot,
    nova_manager
) -> bool:
    """
    Handle disruptor nova firing and fallback safety.
    
    Attempts to fire nova, falls back to KeepUnitSafe if not fired.
    
    Args:
        disruptor: The Disruptor unit to control
        enemies: Enemy units (filtered for disruptor targeting)
        friendly_units: Friendly units (to avoid friendly fire)
        avoid_grid: Avoidance grid for safety
        bot: Bot instance (for use_disruptor_nova access)
        nova_manager: NovaManager instance for coordination
        
    Returns:
        True if nova fired or behavior registered, False otherwise
    """
    from sc2.ids.ability_id import AbilityId
    
    if AbilityId.EFFECT_PURIFICATIONNOVA in disruptor.abilities:
        try:
            result = bot.use_disruptor_nova.execute(
                disruptor, enemies, friendly_units, nova_manager
            )
            if not result:  # Nova didn't fire - keep safe
                bot.register_behavior(KeepUnitSafe(disruptor, avoid_grid))
            return True
        except Exception as e:
            print(f"DEBUG ERROR in disruptor handling: {e}")
            bot.register_behavior(KeepUnitSafe(disruptor, avoid_grid))
            return True
    else:
        # Ability not ready - keep safe
        bot.register_behavior(KeepUnitSafe(disruptor, avoid_grid))
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


