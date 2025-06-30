# bot/hub/reactions.py
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
    #TODO if there are no enemy within our main base i.e they left return to mining
    #TODO once all enemies a dead return to mining - check flags
    # Get all enemy units in our base and filter for workers
    enemy_units = bot.mediator.get_units_in_range(
        start_points=[bot.start_location],
        distances=20,  # Larger radius to catch workers coming in
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
