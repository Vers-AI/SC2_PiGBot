# bot/managers/reactions.py
import numpy as np

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.units import Units
from sc2.position import Point2


# Ares imports
from ares.consts import UnitRole, UnitTreeQueryType
from ares.behaviors.combat.individual import PathUnitToTarget, WorkerKiteBack
from ares.behaviors.combat import CombatManeuver
from ares.managers.manager_mediator import ManagerMediator
from ares.dicts.unit_data import UNIT_DATA

from cython_extensions import (
    cy_distance_to, cy_distance_to_squared, cy_center, cy_find_units_center_mass
)



def defend_cannon_rush(bot):
    """
    Defends against cannon rush by pulling appropriate number of workers.
    Manages bot state flags to coordinate with other threat responses.
    
    Args:
        bot: The bot instance
        enemy_probes: Enemy probe units involved in cannon rush
        enemy_cannons: Enemy cannons (in progress or completed)
    """
    # Only respond if we haven't completed the cannon rush response
    enemy_units: Units = bot.mediator.get_units_in_range(
                        start_points=[bot.start_location],
                        distances=14,
                        query_tree=UnitTreeQueryType.AllEnemy,
                    )[0]
    if not getattr(bot, '_cannon_rush_completed', False):
        # Set initial flags if not already set
        if not getattr(bot, '_cannon_rush_active', False):
            bot._cannon_rush_active = True
            bot.build_order_runner.switch_opening("Cheese_Reaction_Build")
            bot._used_cheese_response = True
            bot._under_attack = True
            bot._worker_cannon_rush_response = True
        
        enemy_probes = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PROBE)
        enemy_cannons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PHOTONCANNON)
        enemy_pylons = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PYLON)

        # Calculate how many workers to pull (1 per cannon + 1 per 2 probes, max 8)
        workers_needed = min(24, len(enemy_cannons) + (len(enemy_probes) // 2) + 8)
        
        # Get current defending workers
        defending_workers = bot.mediator.get_units_from_role(
            role=UnitRole.DEFENDING,
            unit_type=UnitTypeId.PROBE
        )
        
        # Get workers that should be mining (not already defending)
        available_workers = bot.workers.filter(
            lambda w: w.tag not in defending_workers.tags
        )
        
        # Assign more workers if needed
        while len(defending_workers) < workers_needed and available_workers:
            worker = available_workers.closest_to(bot.start_location)
            if not worker:
                break
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
            defending_workers.append(worker)
            available_workers.remove(worker)
        
        # Target selection and attack logic
        for worker in defending_workers:
            # Prioritize cannons that are nearly complete or complete
            urgent_targets = enemy_cannons.filter(
                lambda c: c.build_progress > 0.5 or c.is_ready
            )
            
            if urgent_targets:
                target = urgent_targets.closest_to(worker)
            elif enemy_probes:
                target = enemy_probes.closest_to(worker)
            elif enemy_cannons:  # Only target cannons < 50% if nothing else
                target = enemy_cannons.closest_to(worker)
            elif enemy_pylons:
                target = enemy_pylons.closest_to(worker)
            else:
                # No targets, return to mineral line
                bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
                
            # Attack the target
            worker.attack(target)
        
        # Check if threat is over
        if not enemy_probes and not enemy_cannons and not enemy_pylons:
            # Small delay before cleaning up to ensure threat is really gone
            if not hasattr(bot, '_cannon_rush_cleanup_timer'):
                bot._cannon_rush_cleanup_timer = bot.time
            elif bot.time - bot._cannon_rush_cleanup_timer > 10:  # 10 second delay
                # Clean up workers and flags
                for worker in defending_workers:
                    bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
                
                # Reset flags
                bot._cannon_rush_completed = True
                bot._cannon_rush_response = False
                bot._used_cheese_response = False
                bot._under_attack = False
                
                # Clean up timers
                if hasattr(bot, '_cannon_rush_cleanup_timer'):
                    del bot._cannon_rush_cleanup_timer
        else:
            # Reset cleanup timer if we see threats again
            if hasattr(bot, '_cannon_rush_cleanup_timer'):
                del bot._cannon_rush_cleanup_timer

def defend_worker_rush(bot):
    """
    Defends against worker rush by pulling appropriate number of workers.
    Manages bot state flags to coordinate with other threat responses.
    
    Args:
        bot: The bot instance
    """
    #TODO Fix why Probes dont' return to work after worker rush
    # Get all enemy units in our base and filter for workers
    
    enemy_units = bot.mediator.get_units_in_range(
        start_points=[bot.natural_expansion if bot.structures.closer_than(8, bot.natural_expansion) else bot.start_location],
        distances=25,  # Larger radius to catch workers coming in
        query_tree=UnitTreeQueryType.AllEnemy,
    )[0]
    enemy_workers = enemy_units.filter(lambda u: u.type_id == UnitTypeId.PROBE)

    # Only respond if we actually see enemy workers
    if not enemy_workers:
        return

    # Set initial flags if not already set
    if not getattr(bot, '_worker_rush_active', False):
        bot._worker_rush_active = True
        bot.build_order_runner.switch_opening("Cheese_Reaction_Build")
        bot._used_cheese_response = True
        bot._under_attack = True
        bot._not_worker_rush = False

    # Get current defending workers
    defending_workers = bot.mediator.get_units_from_role(
        role=UnitRole.DEFENDING,
        unit_type=UnitTypeId.PROBE
    )
    
    # Calculate how many workers to pull (1.5x enemy workers, min 4, max 16)
    workers_needed = min(16, max(4, int(len(enemy_workers) * 1.5)))
    
    # Get workers that should be mining (not already defending)
    available_workers = bot.workers.filter(
        lambda w: w.tag not in defending_workers.tags
    )
    
    # Assign more workers if needed
    while len(defending_workers) < workers_needed and available_workers:
        worker = available_workers.closest_to(bot.start_location)
        if not worker:
            break
        bot.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
        defending_workers.append(worker)
        available_workers.remove(worker)
    
    # Target selection and attack logic
    for worker in defending_workers:
        # Find closest enemy worker to this worker
        if enemy_workers:
            target = enemy_workers.closest_to(worker)
            # Use WorkerKiteBack behavior for better micro
            bot.register_behavior(WorkerKiteBack(unit=worker, target=target))
        else:
            # No targets, return to mining
            if bot.mineral_field:
                mf = bot.mineral_field.closest_to(bot.start_location)
                worker.gather(mf)
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
    
    # Check if threat is over (no enemy workers for 5 seconds)
    if not enemy_workers:
        if not hasattr(bot, '_worker_rush_cleanup_timer'):
            bot._worker_rush_cleanup_timer = bot.time
        elif bot.time - bot._worker_rush_cleanup_timer > 5.0:  # 5 second delay
            # Clean up workers by returning them to gathering
            for worker in defending_workers:
                bot.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
            
            # Reset flags
            bot._worker_rush_active = False
            bot._not_worker_rush = True
            bot._used_cheese_response = False
            bot._under_attack = False
            
            # Clean up timer
            if hasattr(bot, '_worker_rush_cleanup_timer'):
                del bot._worker_rush_cleanup_timer
    else:
        # Reset cleanup timer if we see threats again
        if hasattr(bot, '_worker_rush_cleanup_timer'):
            del bot._worker_rush_cleanup_timer


def cheese_reaction(bot):
    """
    Builds pylon/gateway/shield battery to defend early cheese.
    """
    #print(f"Current build: {bot.build_order_runner.chosen_opening}")
    bot.build_order_runner.switch_opening("Cheese_Reaction_Build")

    # Cancel a fast-expanding Nexus if it's started and we detect cheese
    pending_townhalls = bot.structure_pending(UnitTypeId.NEXUS)
    if pending_townhalls == 1 and bot.time < 2 * 60:
        for pt in bot.townhalls.not_ready:
            bot.mediator.cancel_structure(structure=pt)

    

def one_base_reaction(bot):
    bot.build_order_runner.switch_opening("One_Base_Reaction_Build")

    # Set the flags for 1 base reaction
    bot._used_one_base_response = True
    if bot.build_order_runner.build_completed:
        bot._one_base_reaction_completed = True

from bot.utilities.intel import get_enemy_cannon_rushed

def early_threat_sensor(bot):
    """
    Detects early threats like zergling rush, proxy zealots, etc.
    Sets flags so the bot can respond (e.g., cheese_reaction).
    """
    if bot.mediator.get_enemy_worker_rushed and bot.game_state == 0:
        print("Rushed worker detected")
        bot._not_worker_rush = False
        bot._used_cheese_response = True
    
    # Check for cannon rush
    elif get_enemy_cannon_rushed(bot):
        print("Cannon rush detected")
        bot._used_cheese_response = True
        bot._cannon_rush_response = True
    
    elif (
        bot.mediator.get_enemy_ling_rushed
        or (bot.mediator.get_enemy_marauder_rush and bot.time < 150.0)
        or bot.mediator.get_enemy_marine_rush
        or bot.mediator.get_is_proxy_zealot
        or bot.mediator.get_enemy_ravager_rush
        or bot.mediator.get_enemy_went_marine_rush
        or bot.mediator.get_enemy_four_gate
        or bot.mediator.get_enemy_roach_rushed
    ):
        bot._used_cheese_response = True
    
    # Scouting for Enemy 1 base build 
    elif 2.5 * 60 < bot.time < 3.5 * 60 and not (bot.mediator.get_enemy_expanded or bot._used_one_base_response):
        # Get enemy natural location
        enemy_natural = bot.mediator.get_enemy_nat
        grid: np.ndarray = bot.mediator.get_ground_grid
        bot._not_worker_rush = True

        # Assign BUILD_RUNNER_SCOUT units to SCOUTING role
        if build_runner_scout_units := bot.mediator.get_units_from_role(
            role=UnitRole.BUILD_RUNNER_SCOUT, unit_type=bot.worker_type
        ):
            bot.mediator.batch_assign_role(
                tags=build_runner_scout_units.tags, role=UnitRole.SCOUTING
            )
        
        # Get scout units with SCOUTING role
        scout_units: Units = bot.mediator.get_units_from_role(
            role=UnitRole.SCOUTING, 
            unit_type=bot.worker_type
        )


        # Check if scout units exist
        if scout_units: 
            # Check if enemy natural is visible
            if bot.is_visible(enemy_natural):
                # Check if enemy has expanded
                if not bot.mediator.get_enemy_expanded:
                    # No expansion detected, trigger one base reaction
                    one_base_reaction(bot)
                    
                    # switch roles back to gathering
                    for scout in scout_units:
                        bot.mediator.switch_roles(
                            from_role=UnitRole.SCOUTING, to_role=UnitRole.GATHERING
                        )
                else:
                    # Enemy has expanded, continue scouting or other logic
                    pass
            else:
                # Enemy natural not visible, path scout to natural
                for scout in scout_units:
                    bot.register_behavior(
                        PathUnitToTarget(
                            unit=scout, 
                            grid=grid,
                            target=enemy_natural
                        )
                    )
        else:
            # If no scout units, grab one worker to scout
            if worker := bot.mediator.select_worker(
                target_position=bot.mediator.get_enemy_nat, force_close=True
            ):
                bot.mediator.assign_role(tag=worker.tag, role=UnitRole.SCOUTING)


# ===== THREAT ASSESSMENT FUNCTIONS =====
# Moved from combat.py for better separation of concerns

def assess_threat_severity(bot, enemy_units: Units, threat_location: Point2) -> dict:
    """
    Enhanced threat assessment that determines both severity and appropriate response.
    Returns dict with threat_level, required_units, and response_type.
    """
    if not enemy_units:
        return {"threat_level": 0, "required_units": 0, "response_type": "none"}
    
    # Calculate base threat value
    threat_value = 0
    unit_count = enemy_units.amount
    
    # Categorize threats by type
    harassment_units = enemy_units.filter(lambda u: u.type_id in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG
    })
    
    combat_units = enemy_units.filter(lambda u: u.type_id not in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG,
        UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE
    })
    
    # Weight different unit types
    for unit in enemy_units:
        if unit.type_id in UNIT_DATA:
            base_weight = UNIT_DATA[unit.type_id]['army_value']
            # Scale by health percentage
            health_factor = (unit.health + unit.shield) / (unit.health_max + unit.shield_max)
            threat_value += base_weight * health_factor
    
    # Distance factor - closer threats are more urgent
    closest_base = min(bot.townhalls, key=lambda th: cy_distance_to_squared(th.position, threat_location))
    distance_to_base = cy_distance_to(threat_location, closest_base.position)
    distance_factor = max(0.5, 2.0 - (distance_to_base / 20.0))
    threat_value *= distance_factor
    
    # Determine response type and required units
    if harassment_units.amount >= unit_count * 0.8 and unit_count <= 6:
        # Mostly harassment units
        response_type = "harassment_response"
        required_units = min(4, max(2, unit_count))
        threat_level = min(3, threat_value)
    elif combat_units.amount > 3 or threat_value > 15:
        # Significant combat threat
        response_type = "combat_response" 
        required_units = max(6, int(threat_value * 0.8))
        threat_level = min(10, threat_value)
    else:
        # Mixed or small threat
        response_type = "patrol_response"
        required_units = min(3, max(1, unit_count // 2))
        threat_level = min(5, threat_value)
    
    return {
        "threat_level": int(threat_level),
        "required_units": required_units,
        "response_type": response_type,
        "harassment_ratio": harassment_units.amount / max(1, unit_count),
        "unit_count": unit_count,
        "threat_value": threat_value
    }


def assess_threat(bot, enemy_units: Units, own_forces: Units, return_details: bool = False):
    """
    Enhanced threat assessment that can return simple score or detailed analysis.
    
    Args:
        return_details: If True, returns dict with detailed analysis. If False, returns int score.
    """
    if not enemy_units:
        return {"threat_level": 0, "required_units": 0, "response_type": "none"} if return_details else 0
    
    # Calculate base threat value
    threat_value = 0
    unit_count = enemy_units.amount
    
    # Check for damage-dealing low threats (e.g., lings attacking probes/buildings)
    damage_bonus = _assess_damage_threat(bot, enemy_units)
    
    # Categorize threats by type  
    harassment_units = enemy_units.filter(lambda u: u.type_id in {
        UnitTypeId.REAPER, UnitTypeId.ADEPT, UnitTypeId.ORACLE,
        UnitTypeId.HELLION, UnitTypeId.BANSHEE, UnitTypeId.LIBERATORAG
    })
    
    # Weight different unit types
    for unit in enemy_units:
        if unit.type_id in UNIT_DATA:
            base_weight = UNIT_DATA[unit.type_id]['army_value']
            # Scale by health percentage
            health_factor = (unit.health + unit.shield) / max(1, unit.health_max + unit.shield_max)
            threat_value += base_weight * health_factor
    
    # Density check: adjust threat level based on enemy clustering
    center = cy_center(enemy_units)
    cluster_count = 0
    for unit in enemy_units:
        if cy_distance_to(unit.position, center) <= 5.0:
            cluster_count += 1
    
    # If fewer than 3 enemy units are clustered, scale down the threat level
    if cluster_count < 3:
        threat_value *= 0.5
    
    # Apply damage bonus for units actively damaging our assets
    threat_value += damage_bonus
    
    # Simple integer return for backward compatibility
    if not return_details:
        return max(round(threat_value), 0)
    
    # Enhanced details for new system
    # (This replaces assess_threat_severity functionality)
    if len(bot.townhalls) > 0:
        closest_base = min(bot.townhalls, key=lambda th: cy_distance_to_squared(th.position, center))
        distance_to_base = cy_distance_to(center, closest_base.position)
        distance_factor = max(0.5, 2.0 - (distance_to_base / 20.0))
        threat_value *= distance_factor
    
    # Apply damage bonus again after distance factor (for detailed analysis)
    threat_value += damage_bonus
    
    # Determine response type and required units
    # Damage-dealing threats get elevated response even if low unit count
    if damage_bonus > 2.0:  # Critical damage being dealt
        response_type = "damage_response"
        required_units = min(3, max(1, unit_count + 1))  # +1 extra for damage threats
        threat_level = min(6, max(3, threat_value))  # Minimum level 3 for damage threats
    elif harassment_units.amount >= unit_count * 0.8 and unit_count <= 6:
        response_type = "harassment_response"
        required_units = min(4, max(2, unit_count))
        threat_level = min(3, threat_value)
    elif unit_count > 3 and threat_value > 10:
        response_type = "combat_response" 
        required_units = max(6, int(threat_value * 0.8))
        threat_level = min(10, threat_value)
    else:
        response_type = "patrol_response"
        required_units = min(3, max(1, unit_count // 2))
        threat_level = min(5, threat_value)
    
    return {
        "threat_level": int(threat_level),
        "required_units": required_units,
        "response_type": response_type,
        "harassment_ratio": harassment_units.amount / max(1, unit_count),
        "unit_count": unit_count,
        "threat_value": threat_value
    }


def _assess_damage_threat(bot, enemy_units: Units) -> float:
    """
    Assess if low-threat units are actively doing damage to our assets.
    Returns bonus threat value for units near damaged/low-health friendly units.
    """
    damage_bonus = 0.0
    
    # Get our damaged units (buildings + workers)
    damaged_buildings = bot.structures.filter(lambda s: s.health_percentage < 1.0)
    damaged_workers = bot.workers.filter(lambda w: w.health_percentage < 1.0)
    
    # Get very low health units that need immediate attention
    critical_buildings = bot.structures.filter(lambda s: s.health_percentage < 0.5)
    critical_workers = bot.workers.filter(lambda w: w.health_percentage < 0.3)
    
    # Check if enemy units are near damaged assets
    for enemy in enemy_units:
        # Higher bonus for units near critical assets
        if critical_buildings:
            closest_critical = critical_buildings.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_critical.position) <= 3.0:
                damage_bonus += 3.0  # High priority for units attacking critical buildings
        
        if critical_workers:
            closest_critical_worker = critical_workers.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_critical_worker.position) <= 2.0:
                damage_bonus += 2.0  # Medium priority for units attacking low-health workers
        
        # Medium bonus for units near damaged assets
        if damaged_buildings:
            closest_damaged = damaged_buildings.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_damaged.position) <= 3.0:
                damage_bonus += 1.5
        
        if damaged_workers:
            closest_damaged_worker = damaged_workers.closest_to(enemy.position)
            if cy_distance_to(enemy.position, closest_damaged_worker.position) <= 2.0:
                damage_bonus += 1.0
    
    return damage_bonus


def allocate_defensive_forces(bot, threat_info: dict, threat_location: Point2, enemy_units: Units) -> Units:
    """
    Smart unit allocation based on threat value and air/ground requirements.
    Uses army_value matching and capability filtering like the examples.
    """
    available_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    
    if not available_units:
        return Units([], bot)
    
    # Dynamic threat composition check (following example patterns)
    has_air = any(unit.is_flying for unit in enemy_units)
    has_ground = any(not unit.is_flying for unit in enemy_units)
    
    # Filter capable units (exactly like examples)
    if has_air and has_ground:
        # Mixed threat - prioritize units that can attack both, fallback to air-capable
        capable_units = available_units.filter(lambda u: u.can_attack_both)
        if not capable_units:
            capable_units = available_units.filter(lambda u: u.can_attack_air)
    elif has_air:
        capable_units = available_units.filter(lambda u: u.can_attack_air)
    else:
        capable_units = available_units.filter(lambda u: u.can_attack_ground)
    
    if not capable_units:
        # Fallback to any available units if no capable units found
        capable_units = available_units
    
    response_type = threat_info["response_type"]
    threat_value = threat_info.get("threat_value", 0)
    
    # For damage-dealing threats, send closest capable units immediately
    if response_type == "damage_response":
        closest_units = capable_units.sorted(lambda u: cy_distance_to_squared(u.position, threat_location))
        required_count = min(threat_info["required_units"], len(closest_units))
        return closest_units[:required_count]
    
    # Value-based allocation for other response types
    selected_units = []
    current_value = 0.0
    target_value = max(threat_value * 1.1, 2.0)  # Slight over-allocation for safety
    
    # Sort by distance with slight preference for higher value units
    sorted_units = capable_units.sorted(lambda u: 
        cy_distance_to_squared(u.position, threat_location) + 
        (UNIT_DATA.get(u.type_id, {}).get('army_value', 1.0) * -0.1)  # Negative for preference
    )
    
    # Select units to match threat value
    for unit in sorted_units:
        if current_value >= target_value and len(selected_units) >= 1:
            break
        if len(selected_units) >= threat_info["required_units"]:
            break
            
        selected_units.append(unit)
        unit_value = UNIT_DATA.get(unit.type_id, {}).get('army_value', 1.0)
        current_value += unit_value
    
    # Ensure minimum response for non-trivial threats
    if not selected_units and capable_units:
        selected_units = [capable_units.closest_to(threat_location)]
    
    return Units(selected_units, bot)


def threat_detection(bot, main_army: Units) -> None:
    """
    Enhanced threat detection with smart force allocation.
    Detects threats near bases and responds with appropriate force levels
    instead of always redirecting the entire main army.
    """
    # Import here to avoid circular dependency
    from bot.managers.combat import control_main_army
    
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases

    if ground_near or flying_near:
        # Combine ground + air threats
        all_threats = {}
        for key, value in ground_near.items():
            all_threats[key] = value.copy()
        for key, value in flying_near.items():
            all_threats.setdefault(key, set()).update(value)

        main_army_should_respond = False
        
        for _, enemy_tags in all_threats.items():
            enemy_units = bot.enemy_units.tags_in(enemy_tags)
            if not enemy_units:
                continue
                
            threat_position, _ = cy_find_units_center_mass(enemy_units, 10.0)
            threat_position = Point2(threat_position)
            
            # Use enhanced threat assessment (now consolidated)
            threat_info = assess_threat(bot, enemy_units, main_army, return_details=True)
            # Type assertion since we know return_details=True returns dict
            assert isinstance(threat_info, dict), "assess_threat with return_details=True should return dict"
            current_threat_level = threat_info["threat_level"]  # For original hysteresis thresholds
            
            # Preserve original hysteresis thresholds for bot._under_attack flag
            if bot.time < 3 * 60 and bot._used_cheese_response:
                high_threshold = 2
                low_threshold = 1
            else:
                high_threshold = 5
                low_threshold = 2
            
            # Update global threat status using original hysteresis thresholds
            if bot._under_attack:
                if current_threat_level < low_threshold:
                    bot._under_attack = False
            else:
                if current_threat_level >= high_threshold:
                    bot._under_attack = True
            
            # Smart force allocation instead of all-or-nothing response
            if bot._under_attack or threat_info["threat_level"] >= 3:
                # Allocate appropriate defensive forces first
                defensive_units = allocate_defensive_forces(bot, threat_info, threat_position, enemy_units)
                
                if defensive_units:
                    # Assign defensive roles and control them
                    for unit in defensive_units:
                        bot.mediator.assign_role(tag=unit.tag, role=UnitRole.DEFENDING)
                    
                    defensive_squads = bot.mediator.get_squads(role=UnitRole.DEFENDING, squad_radius=6.0)
                    if defensive_squads:
                        control_main_army(bot, defensive_units, threat_position, defensive_squads)
                    else:
                        # Fallback: direct unit commands when squads aren't available
                        for unit in defensive_units:
                            unit.attack(threat_position)
                
                # Only redirect main army for overwhelming threats (preserving critical defense)
                if (threat_info["threat_level"] >= 8 or 
                    (threat_info["response_type"] == "combat_response" and threat_info["unit_count"] > 8) or
                    current_threat_level >= high_threshold * 2):  # Very high threat by original standards
                    main_army_should_respond = True
                    control_main_army(bot, main_army, threat_position, 
                                    bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
        
        # Store whether main army responded for other systems to check
        bot._main_army_defending = main_army_should_respond
