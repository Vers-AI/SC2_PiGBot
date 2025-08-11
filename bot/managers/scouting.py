# scouting

import numpy as np
from typing import Dict, List, Optional, Set
from sc2.data import AbilityId
from sc2.units import Units
from sc2.position import Point2
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import KeepUnitSafe, PathUnitToTarget, UseAbility
from ares.consts import UnitRole, UnitTreeQueryType
from bot.managers.combat import attack_target

# TODO add survilaence mode to primary obs when its at th ol spot, and patrol obs when it reaches desitnation 
def control_observers(bot, all_observers: Units, main_army: Units) -> None:
    """
    Coordinate multiple observers with different roles, adapting to the game situation.
    
    Parameters:
    - bot: The bot instance
    - all_observers: All observer units
    - main_army: The main army units
    """
    # Quick check if we have any observers
    if not all_observers:
        return
    
    # Get observers by their assignment
    primary_observer = None
    if bot.observer_assignments.get("primary"):
        primary_candidates = all_observers.filter(lambda o: o.tag == bot.observer_assignments.get("primary"))
        primary_observer = primary_candidates.first if primary_candidates else None
    
    army_observer = None
    if bot.observer_assignments.get("army"):
        army_candidates = all_observers.filter(lambda o: o.tag == bot.observer_assignments.get("army"))
        army_observer = army_candidates.first if army_candidates else None
    
    patrol_observers = Units([], bot._game_data)
    if bot.observer_assignments.get("patrol"):
        patrol_observers = all_observers.filter(lambda o: o.tag in bot.observer_assignments.get("patrol", []))
    
    # Control each type of observer
    control_primary_observer(bot, primary_observer, main_army)
    control_army_observer(bot, army_observer, main_army)
    control_patrol_observers(bot, patrol_observers)


def control_primary_observer(bot, observer: Optional[Unit], main_army: Units) -> None:
    """
    Control the primary observer with adaptive behavior based on game state.
    
    Parameters:
    - bot: The bot instance
    - observer: The primary observer unit
    - main_army: The main army units for potential following
    """
    if not observer:
        return
        
    actions = CombatManeuver()
    air_grid = bot.mediator.get_air_grid
    
    # If we're attacking/under attack AND this is our only observer, it follows the army
    if (bot._commenced_attack or bot._under_attack) and bot.units(UnitTypeId.OBSERVER).amount == 1:
        # Use the army following logic with dynamic lead distance
        if main_army:
            target_point = attack_target(bot, main_army.center)
            
            # Dynamic lead distance based on threat level
            lead_distance = 12
            tentative_target = Point2(main_army.center.towards(target_point, lead_distance))
            
            try:
                influence = air_grid[int(tentative_target.x)][int(tentative_target.y)]
                if influence > 1:
                    lead_distance = 6
            except Exception:
                lead_distance = 10
                
            follow_target = Point2(main_army.center.towards(target_point, lead_distance))
            
            actions.add(PathUnitToTarget(
                unit=observer,
                target=follow_target,
                grid=air_grid
            ))
        else:
            actions.add(PathUnitToTarget(
                unit=observer,
                target=bot.start_location,
                grid=air_grid,
                danger_distance=15
            ))
    # Otherwise, follow its normal assignment based on game phase
    else:
        # Early game: patrol expansions
        if bot.game_state == 0:  # early game
            # Create a list of potential scout targets
            targets = bot.expansion_locations_list[:5] + [bot.enemy_start_locations[0]]
            
            # If we haven't assigned a current target, do so
            tag = observer.tag
            if tag not in bot.observer_targets or not bot.observer_targets[tag]:
                if targets:
                    bot.observer_targets[tag] = targets[0]
            
            # If damaged, run away
            if observer.shield_percentage < 1:
                actions.add(KeepUnitSafe(
                    unit=observer,
                    grid=air_grid
                ))
            else:
                # If close to the current target, pick the next
                current_target = bot.observer_targets.get(tag)
                if current_target and observer.distance_to(current_target) < 1:
                    # Move to the next index
                    current_index = targets.index(current_target)
                    if current_index + 1 < len(targets):
                        bot.observer_targets[tag] = targets[current_index + 1]
                    else:
                        # If we've used all targets, reset to the first one
                        bot.observer_targets[tag] = targets[0]
                
                # Move to the current target if it exists
                if bot.observer_targets.get(tag) is not None:
                    actions.add(PathUnitToTarget(
                        unit=observer,
                        target=bot.observer_targets[tag],
                        grid=air_grid,
                        danger_distance=10
                    ))
        else:
            # Mid/late game: position at enemy natural
            ol_spot = bot.mediator.get_ol_spot_near_enemy_nat
            if ol_spot:
                actions.add(PathUnitToTarget(
                    unit=observer,
                    target=ol_spot,
                    grid=air_grid,
                    danger_distance=15
                ))
                if observer.position.distance_to(ol_spot) < 1:
                    observer(AbilityId.MORPH_SURVEILLANCEMODE)
            else:
                # Fallback if no overlord spot defined
                actions.add(PathUnitToTarget(
                    unit=observer,
                    target=bot.enemy_start_locations[0],
                    grid=air_grid,
                    danger_distance=15
                ))
    
    bot.register_behavior(actions)


def control_army_observer(bot, observer: Optional[Unit], main_army: Units) -> None:
    """
    Control the army observer to follow and provide vision for the main army.
    Prioritizes moving towards cloaked/burrowed enemies near the army if present.
    
    Parameters:
    - bot: The bot instance
    - observer: The army observer unit
    - main_army: The main army units to follow
    """
    if not observer:
        return
    
    actions = CombatManeuver()
    air_grid = bot.mediator.get_air_grid
    
    # Keep the observer safe if it's taking damage
    if observer.shield_percentage < 1:
        actions.add(KeepUnitSafe(
            unit=observer,
            grid=air_grid
        ))
    elif main_army:
        # Check for cloaked/burrowed enemies near the army that need detection
        army_center = main_army.center
        army_threat_radius = 15  # Range to check for enemies near army
        
        # Get enemies near the army
        nearby_enemies = bot.enemy_units.closer_than(army_threat_radius, army_center)
        
        # Filter for enemies that are cloaked or burrowed (excluding drones and observer siege mode)
        priority_enemies = nearby_enemies.filter(lambda u: 
            u.is_cloaked or 
            u.is_burrowed or
            u.type_id in {
                UnitTypeId.LURKERMPBURROWED,
                UnitTypeId.ROACHBURROWED, UnitTypeId.ZERGLINGBURROWED,
                UnitTypeId.HYDRALISKBURROWED, UnitTypeId.ULTRALISKBURROWED,
                UnitTypeId.WIDOWMINEBURROWED
            }
        )
        
        if priority_enemies:
            # Priority: move towards the closest cloaked/burrowed enemy
            closest_priority_enemy = priority_enemies.closest_to(army_center)
            follow_target = closest_priority_enemy.position
        else:
            # Normal behavior: look ahead of the army
            target_point = attack_target(bot, main_army.center)
            
            # Dynamic lead distance based on threat level
            lead_distance = 12
            tentative_target = Point2(main_army.center.towards(target_point, lead_distance))
            
            try:
                # Grid is indexed in (x, y) order according to SC2 coordinate convention
                influence = air_grid[int(tentative_target.x)][int(tentative_target.y)]
                # If influence > 1 (enemy threat), shorten the lead distance to stay safer
                if influence > 1:
                    lead_distance = 6
            except Exception:
                # Fallback in case of index error or grid issue
                lead_distance = 10
            
            # Final follow target ahead of army taking into account the adjusted lead distance
            follow_target = Point2(main_army.center.towards(target_point, lead_distance))
        
        actions.add(PathUnitToTarget(
            unit=observer,
            target=follow_target,
            grid=air_grid,
            sense_danger=False
        ))
    else:
        # No army to follow, stay near main base
        actions.add(PathUnitToTarget(
            unit=observer,
            target=bot.start_location,
            grid=air_grid,
            danger_distance=15
        ))
    
    bot.register_behavior(actions)


def control_patrol_observers(bot, observers: Units) -> None:
    """
    Control additional observers to patrol key map positions.
    
    Parameters:
    - bot: The bot instance
    - observers: The patrol observer units
    """
    if not observers:
        return
    
    actions = CombatManeuver()
    air_grid = bot.mediator.get_air_grid
    
    # Get overlord spots from mediator
    ol_spots = bot.mediator.get_ol_spots
    if not ol_spots or len(ol_spots) == 0:
        # Fallback to expansions if no overlord spots
        ol_spots = bot.expansion_locations_list[5:10] if len(bot.expansion_locations_list) > 5 else []
    
    # Filter out the spot near enemy natural
    enemy_nat_spot = bot.mediator.get_ol_spot_near_enemy_nat
    if enemy_nat_spot and enemy_nat_spot in ol_spots:
        ol_spots.remove(enemy_nat_spot)
    
    # If we still have valid patrol spots
    if ol_spots:
        for i, observer in enumerate(observers):
            # Keep the observer safe if it's taking damage
            if observer.shield_percentage < 1:
                actions.add(KeepUnitSafe(
                    unit=observer,
                    grid=air_grid
                ))
            else:
                tag = observer.tag
                # Assign a spot if not already assigned
                if tag not in bot.observer_targets or not bot.observer_targets[tag]:
                    # Distribute observers evenly across spots
                    spot_index = i % len(ol_spots)
                    bot.observer_targets[tag] = ol_spots[spot_index]
                
                # If close to target, cycle to next spot
                if observer.distance_to(bot.observer_targets[tag]) < 1:
                    current_index = ol_spots.index(bot.observer_targets[tag]) if bot.observer_targets[tag] in ol_spots else 0
                    next_index = (current_index + 1) % len(ol_spots)
                    bot.observer_targets[tag] = ol_spots[next_index]
                
                # Move to assigned target
                actions.add(PathUnitToTarget(
                    unit=observer,
                    target=bot.observer_targets[tag],
                    grid=air_grid,
                    danger_distance=12
                ))
    
    bot.register_behavior(actions)


# Legacy function for backward compatibility
def control_scout(bot, scout_units: Units, main_army: Units) -> None:
    """
    Controls your scouting units. This now redirects to the new observer control system.
    """
    # Filter for only observer units
    observers = scout_units.filter(lambda u: u.type_id == UnitTypeId.OBSERVER)
    
    # Use the new observer control system
    control_observers(bot, observers, main_army)
