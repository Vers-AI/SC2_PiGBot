# bot/hub/reactions.py
import numpy as np

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.units import Units
from sc2.position import Point2


# Ares imports
from ares.consts import UnitRole
from ares.behaviors.combat.individual import PathUnitToTarget, UseAbility
from ares.behaviors.combat import CombatManeuver
from ares.managers.manager_mediator import ManagerMediator


def defend_worker_cannon_rush(bot, enemy_probes, enemy_cannons):
    """
    Defends a cannon rush by pulling worker(s) to attack 
    enemy probes/cannons quickly.
    """
    # End the primary build order
    bot.build_order_runner.set_build_completed()
    bot._used_cheese_response = True
    bot._under_attack = True

    # Select a worker
    worker = bot.mediator.select_worker(target_position=bot.start_location)
    if worker:
        bot.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)

    # Retrieve defending workers
    defending_workers: Units = bot.mediator.get_units_from_role(
        role=UnitRole.DEFENDING, 
        unit_type=UnitTypeId.PROBE
    )

    # Attack enemy probes
    for probe in enemy_probes:
        defending_worker = defending_workers.closest_to(probe)
        if defending_worker:
            defending_worker.attack(probe)

    # Attack enemy cannons
    for cannon in enemy_cannons:
        defending_worker = defending_workers.closest_to(cannon)
        if defending_worker:
            defending_worker.attack(cannon)

def cheese_reaction(bot):
    """
    Builds pylon/gateway/shield battery to defend early cheese.
    """
    print(f"Current build: {bot.build_order_runner.chosen_opening}")
    bot.build_order_runner.switch_opening("Cheese_Reaction_Build")

    # Cancel a fast-expanding Nexus if it's started and we detect cheese
    pending_townhalls = bot.structure_pending(UnitTypeId.NEXUS)
    if pending_townhalls == 1 and bot.time < 2 * 60:
        for pt in bot.townhalls.not_ready:
            bot.mediator.cancel_structure(structure=pt)

    # Set the flag if the build order is completed
    if bot.build_order_runner.build_completed:
        bot._cheese_reaction_completed = True

def one_base_reaction(bot):
    bot.build_order_runner.switch_opening("One_Base_Reaction_Build")

    # Set the flags for 1 base reaction
    bot._used_one_base_response = True
    if bot.build_order_runner.build_completed:
        bot._one_base_reaction_completed = True

def early_threat_sensor(bot):
    """
    Detects early threats like zergling rush, proxy zealots, etc.
    Sets flags so the bot can respond (e.g., cheese_reaction).
    """
    if bot.mediator.get_enemy_worker_rushed:
        print("Rushed worker detected")

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
    elif bot.time > 3.30 * 60 and not bot._used_one_base_response:
        # Get enemy natural location
        enemy_natural = bot.mediator.get_enemy_nat
        grid: np.ndarray = bot.mediator.get_ground_grid

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
