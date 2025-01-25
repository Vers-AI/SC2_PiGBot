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
        bot._used_cheese_response = True

def one_base_reaction(bot):
    bot.build_order_runner.switch_opening("One_Base_Reaction_Build")

    # Set the flag if the build order is completed
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
    
    # Probe Enemy Natural Scout
    elif bot.time > 2 * 60:
        # Get enemy natural location
        enemy_natural = bot.mediator.get_enemy_nat
        grid: np.ndarray = bot.mediator.get_ground_grid
        scout_maneuver : CombatManeuver = CombatManeuver()

        # Retrieve scout units from build runner
        scout_units: Units = bot.mediator.get_units_from_role(
            role=UnitRole.BUILD_RUNNER_SCOUT, 
            unit_type=bot.worker_type
        )
        print(f"Scout unit(s): {scout_units}")
        
        # Only proceed if scout units exist
        if scout_units:
            # Debug print for enemy natural visibility
            print(f"Enemy Natural at {enemy_natural}, Visible: {bot.is_visible(enemy_natural)}")
            
            # Check if enemy natural is visible
            if bot.is_visible(enemy_natural):
                # If enemy natural is visible and hasn't expanded
                if not bot.mediator.get_enemy_expanded:
                    for scout in scout_units:
                        # If before 3:30, path to and hold at enemy natural
                        if bot.time <= 3.5 * 60:
                            # Path to enemy natural
                            scout_maneuver.add(
                                PathUnitToTarget(
                                unit=scout, 
                                grid=grid,
                                target=enemy_natural
                                )
                            )
                            
                            scout_maneuver.add(
                                UseAbility(
                                    unit=scout,
                                    ability=AbilityId.HOLDPOSITION
                                )
                            )
                            bot.register_behavior(scout_maneuver)
                            print(f"Scout registered to path to enemy natural {enemy_natural}")
                    
                    # Check at 3:30 if still no expansion
                    if bot.time > 3.5 * 60:
                        if not bot.mediator.get_enemy_expanded:
                            bot._used_one_base_response = True
                            print("Enemy hasn't expanded by 3:30, triggering one-base reaction")
                        
                        # Always remove scout role at 3:30
                        for scout in scout_units:
                            bot.mediator.clear_unit_role(scout.tag, UnitRole.BUILD_RUNNER_SCOUT)
            else:
                # Enemy natural not visible, move scout to explore
                for scout in scout_units:
                    # If before 3:30, path to enemy natural
                    if bot.time <= 3.5 * 60:
                        scout_maneuver.add(
                                PathUnitToTarget(
                                unit=scout, 
                                grid=grid,
                                target=enemy_natural
                            )
                        )
                        
                        scout_maneuver.add(
                            UseAbility(
                                unit=scout,
                                ability=AbilityId.HOLDPOSITION
                            )
                        )
                        bot.register_behavior(scout_maneuver)
                        print(f"Scout pathing to enemy natural at {enemy_natural}")
