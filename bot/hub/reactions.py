# bot/hub/reactions.py

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.units import Units
from sc2.position import Point2

# Ares imports
from ares.behaviors.macro import BuildStructure
from ares.consts import UnitRole

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
    Handles your 'cheese_reaction' logic from the old code.
    Builds pylon/gateway/shield battery to defend early cheese.
    """
    pylon_count = bot.structures(UnitTypeId.PYLON).amount + bot.structure_pending(UnitTypeId.PYLON)
    gateway_count = bot.structures(UnitTypeId.GATEWAY).amount + bot.structure_pending(UnitTypeId.GATEWAY)
    zealot_count = bot.units(UnitTypeId.ZEALOT).amount
    has_shield_battery = (
        bot.structures(UnitTypeId.SHIELDBATTERY).ready
        or bot.structure_pending(UnitTypeId.SHIELDBATTERY)
    )

    natural = bot.natural_expansion.towards(bot.game_info.map_center, 1)
    pending_townhalls = bot.structure_pending(UnitTypeId.NEXUS)
    cyb = bot.structures(UnitTypeId.CYBERNETICSCORE).ready

    # Cancel a fast-expanding Nexus if it's started and we detect cheese
    if pending_townhalls == 1 and bot.time < 2 * 60:
        for pt in bot.townhalls.not_ready:
            bot.mediator.cancel_structure(structure=pt)

    # Build pylon (up to 2), then gateway, zealots, shield battery, etc.
    if pylon_count < 2:
        if not bot.structure_pending(UnitTypeId.PYLON):
            if bot.can_afford(UnitTypeId.PYLON):
                bot.register_behavior(BuildStructure(
                    base_location=natural, 
                    structure_id=UnitTypeId.PYLON, 
                    closest_to=bot.game_info.map_center
                ))

    if bot.structures(UnitTypeId.PYLON).ready and gateway_count < 2:
        if bot.can_afford(UnitTypeId.GATEWAY) and bot.structure_pending(UnitTypeId.GATEWAY) == 0:
            bot.register_behavior(BuildStructure(
                base_location=natural, 
                structure_id=UnitTypeId.GATEWAY, 
                closest_to=bot.game_info.map_center
            ))

    if gateway_count > 0 and zealot_count < 2:
        if bot.can_afford(UnitTypeId.ZEALOT):
            bot.train(UnitTypeId.ZEALOT, closest_to=natural)
            # Chrono gate if possible
            for th in bot.townhalls:
                if th.energy >= 50:
                    gateways = [
                        g for g in bot.mediator.get_own_structures_dict[UnitTypeId.GATEWAY]
                        if g.build_progress >= 1.0 and not g.is_idle
                    ]
                    if gateways:
                        th(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, gateways[0])

    if gateway_count > 0 and zealot_count >= 1:
        if not bot.structures(UnitTypeId.SHIELDBATTERY).ready and cyb:
            if bot.can_afford(UnitTypeId.SHIELDBATTERY) and bot.structure_pending(UnitTypeId.SHIELDBATTERY) == 0:
                bot.register_behavior(BuildStructure(
                    base_location=natural, 
                    structure_id=UnitTypeId.SHIELDBATTERY, 
                    closest_to=bot.game_info.map_center
                ))
        if cyb:
            # Warpgate research
            if bot.can_afford(AbilityId.RESEARCH_WARPGATE):
                cyb.first(AbilityId.RESEARCH_WARPGATE)
            # Chrono the Cybercore
            for th in bot.townhalls:
                if th.energy >= 50:
                    for ccore in bot.mediator.get_own_structures_dict[UnitTypeId.CYBERNETICSCORE]:
                        if ccore.build_progress >= 1.0 and not ccore.is_idle:
                            th(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, ccore)

    if has_shield_battery and zealot_count >= 2 and pylon_count < 3:
        if bot.can_afford(UnitTypeId.PYLON) and bot.structure_pending(UnitTypeId.PYLON) == 0:
            bot.register_behavior(BuildStructure(
                base_location=natural, 
                structure_id=UnitTypeId.PYLON, 
                closest_to=bot.game_info.map_center
            ))
            bot._cheese_reaction_completed = True

def one_base_reaction(bot):
    """
    If the enemy is staying on one base, build extra defenses 
    and hold off expansions for a while.
    """
    pylon_count = bot.structures(UnitTypeId.PYLON).amount + bot.structure_pending(UnitTypeId.PYLON)
    gateway_count = bot.structures(UnitTypeId.GATEWAY).amount + bot.structure_pending(UnitTypeId.GATEWAY)
    stalker_count = bot.units(UnitTypeId.STALKER).amount
    shield_battery_count = (
        bot.structures(UnitTypeId.SHIELDBATTERY).amount 
        + bot.structure_pending(UnitTypeId.SHIELDBATTERY)
    )

    natural = bot.natural_expansion.towards(bot.game_info.map_center, 1)
    cyb = bot.structures(UnitTypeId.CYBERNETICSCORE).ready

    if pylon_count < 2:
        if not bot.structure_pending(UnitTypeId.PYLON):
            if bot.can_afford(UnitTypeId.PYLON):
                bot.register_behavior(BuildStructure(
                    base_location=natural, 
                    structure_id=UnitTypeId.PYLON, 
                    closest_to=bot.game_info.map_center
                ))

    # Once we have a gateway and 3+ stalkers, spam shield batteries
    if gateway_count > 0 and stalker_count >= 3:
        if shield_battery_count < 2:
            if cyb and bot.can_afford(UnitTypeId.SHIELDBATTERY):
                # Build shield battery near natural
                if bot.structures(UnitTypeId.SHIELDBATTERY).closer_than(8, natural).amount == 0:
                    bot.register_behavior(BuildStructure(
                        base_location=natural, 
                        structure_id=UnitTypeId.SHIELDBATTERY, 
                        closest_to=bot.game_info.map_center
                    ))
                # Possibly one at the main if you want extra defense
                elif bot.structures(UnitTypeId.SHIELDBATTERY).closer_than(8, bot.start_location).amount == 0:
                    bot.register_behavior(BuildStructure(
                        base_location=bot.start_location, 
                        structure_id=UnitTypeId.SHIELDBATTERY, 
                        closest_to=bot.townhalls[0].position.towards(bot.start_location, -1)
                    ))

        if shield_battery_count >= 2:
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

    elif bot.time > 2 * 60 and not bot.mediator.get_enemy_expanded:
        bot._used_one_base_response = True
