# bot/hub/macro.py

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.units import Units

from ares.behaviors.macro import (
    ProductionController,
    SpawnController,
    MacroPlan,
    AutoSupply,
    BuildStructure,
)
from ares.consts import UnitRole

from bot.hub.reactions import one_base_reaction
from bot.hub.scouting import control_scout

# Army composition constants (instead of @property on a class)
STANDARD_ARMY = {
    UnitTypeId.IMMORTAL: {"proportion": 0.2, "priority": 2},
    UnitTypeId.COLOSSUS: {"proportion": 0.1, "priority": 3},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.45, "priority": 1},
    UnitTypeId.ZEALOT: {"proportion": 0.25, "priority": 0},
}

CHEESE_DEFENSE_ARMY = {
    UnitTypeId.ZEALOT: {"proportion": 0.5, "priority": 0},
    UnitTypeId.STALKER: {"proportion": 0.4, "priority": 1},
    UnitTypeId.ADEPT: {"proportion": 0.1, "priority": 2},
}


async def handle_macro(
    bot,
    iteration: int,
    main_army: Units,
    warp_prism: Units,
    scout_units: Units,
    freeflow: bool,
) -> None:
    """
    Main macro logic: builds units, keeps supply up, reacts to cheese, 
    or transitions to late game. Call this in your bot's on_step.
    """
    # If our build is done and we haven't detected cheese, do standard macro
    if bot.build_order_runner.build_completed and not bot._used_cheese_response:
        macro_plan = MacroPlan()
        macro_plan.add(AutoSupply(base_location=bot.start_location))
        macro_plan.add(ProductionController(STANDARD_ARMY, base_location=bot.start_location))

        # Spawn units near Warp Prism if available, else at base
        if warp_prism:
            prism_position = warp_prism[0].position
            macro_plan.add(
                SpawnController(STANDARD_ARMY, spawn_target=prism_position, freeflow_mode=freeflow)
            )
        else:
            macro_plan.add(SpawnController(STANDARD_ARMY, freeflow_mode=freeflow))

        bot.register_behavior(macro_plan)

    # If we detected cheese
    elif bot._used_cheese_response:
        bot.cheese_reaction()  # Example: a function from your reactions module

        if not bot.build_order_runner.build_completed:
            bot.build_order_runner.set_build_completed()

        # If we finished initial cheese response, maybe expand or produce a defense plan
        if bot._cheese_reaction_completed:
            if not bot._under_attack:
                if bot.townhalls.ready.amount <= 1 and bot.structure_pending(UnitTypeId.NEXUS) == 0:
                    if bot.can_afford(UnitTypeId.NEXUS):
                        await bot.expand_to_next_location()
                elif bot.townhalls.ready.amount <= 2 and bot.structure_pending(UnitTypeId.NEXUS) == 0:
                    if bot.can_afford(UnitTypeId.NEXUS):
                        await bot.expand_to_next_location()

            # Build a cheese defense plan
            cheese_defense_plan = MacroPlan()
            cheese_defense_plan.add(AutoSupply(base_location=bot.start_location))
            cheese_defense_plan.add(
                SpawnController(CHEESE_DEFENSE_ARMY, spawn_target=bot.start_location, freeflow_mode=freeflow)
            )
            cheese_defense_plan.add(
                ProductionController(CHEESE_DEFENSE_ARMY, base_location=bot.start_location)
            )

            bot.register_behavior(cheese_defense_plan)

    # If we detected a one-base play
    if bot._used_one_base_response:
        one_base_reaction(bot)


    # Build extra probes
    if bot._used_cheese_response and bot.townhalls.ready.amount <= 2 and bot.workers.amount < 44:
        if bot.can_afford(UnitTypeId.PROBE):
            bot.train(UnitTypeId.PROBE)
        _chrono_townhalls(bot)
    elif bot.townhalls.ready.amount == 3 and bot.workers.amount < 66:
        if bot.can_afford(UnitTypeId.PROBE):
            bot.train(UnitTypeId.PROBE)
        _chrono_townhalls(bot)

    # If something went horribly wrong with the build order, mark it as complete
    if bot.minerals > 2500 and not bot.build_order_runner.build_completed:
        bot.build_order_runner.set_build_completed()

    # Scout control or build observer if no scout
    if scout_units and main_army:
        control_scout(bot, scout_units, main_army)
    else:
        if bot.time > 4 * 60:
            if bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
                if (bot.units(UnitTypeId.OBSERVER).amount < 1 
                    and bot.already_pending(UnitTypeId.OBSERVER) == 0
                    and bot.can_afford(UnitTypeId.OBSERVER)):
                    bot.train(UnitTypeId.OBSERVER)

    # Warp Prism follows main army
    if warp_prism:
        bot.warp_prism_follower(warp_prism, main_army)

    # Merge Archons if we have at least 2 High Templars
    if bot.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
        for templar in bot.units(UnitTypeId.HIGHTEMPLAR).ready:
            templar(AbilityId.MORPH_ARCHON)


def worker_production(bot) -> None:
    """
    Simple separate function for worker production logic if desired.
    Could be merged into handle_macro if you like.
    """
    # Example check:
    if bot.townhalls.ready and bot.workers.amount < 70:
        if bot.can_afford(UnitTypeId.PROBE):
            bot.train(UnitTypeId.PROBE)


def _chrono_townhalls(bot) -> None:
    """
    Helper function to chrono your townhalls if possible.
    """
    for th in bot.townhalls:
        if not th.is_idle and th.energy >= 50:
            th(AbilityId.EFFECT_CHRONOBOOST, th)
