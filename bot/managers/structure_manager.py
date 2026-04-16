# bot/managers/structure_manager.py
# Purpose: Nexus ability management (Chronoboost, Energy Recharge, Mass Recall)
# Key Decisions: All Nexus-cast abilities consolidated here for consistent energy management
# Limitations: Mass Recall combat-trigger logic remains in combat.py

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2
from sc2.units import Units

from bot.constants import (
    MASS_RECALL_COOLDOWN,
    MASS_RECALL_ENERGY_COST,
)


def _get_recall_nexus_tag(bot) -> int | None:
    """Return the tag of the Nexus designated to preserve energy for Mass Recall.

    Only designates a recall Nexus once we've expanded (2+ Nexuses), since
    the first Nexus needs to spend freely early on. Uses the main/starting
    Nexus so it preserves 50 energy for emergency recall.
    Returns None if we haven't expanded yet or no suitable Nexus exists.
    """
    # Only designate a recall Nexus after expansion
    if len(bot.townhalls.ready) < 2:
        return None
    for nexus in bot.townhalls.ready:
        if nexus.type_id == UnitTypeId.NEXUS and nexus.position == bot.start_location:
            return nexus.tag
    # Fallback: first available Nexus
    for nexus in bot.townhalls.ready:
        if nexus.type_id == UnitTypeId.NEXUS:
            return nexus.tag
    return None


def use_recharge(bot, main_army: Units) -> bool:
    """
    Uses Energy Recharge on the unit with the lowest energy percentage within range of the closest Nexus.

    Targets energy-using units in priority order: Shield Battery, Mothership, High Templar, Oracle, and Sentry.

    Args:
        bot: The bot instance
        main_army: Units to consider for the army center calculation

    Returns:
        bool: True if Energy Recharge was used, False otherwise
    """
    if not main_army:
        return False

    # Priority order: Shield Battery > Mothership > High Templar > Oracle > Sentry
    priority_unit_types = [
        UnitTypeId.SHIELDBATTERY,
        UnitTypeId.MOTHERSHIP,
        UnitTypeId.HIGHTEMPLAR,
        UnitTypeId.ORACLE,
        UnitTypeId.SENTRY
    ]

    # Find the closest Nexus to the army center
    # Recall Nexus requires 100 energy (50 reserve + 50 cost) to spend
    recall_nexus_tag = _get_recall_nexus_tag(bot)
    closest_nexus = None
    closest_distance = float("inf")
    for nexus in bot.structures(UnitTypeId.NEXUS).ready:
        min_energy = 100 if nexus.tag == recall_nexus_tag else 50
        if nexus.energy < min_energy:
            continue
        distance = main_army.center.distance_to(nexus.position)
        if distance < closest_distance:
            closest_distance = distance
            closest_nexus = nexus

    if not closest_nexus:
        return False

    # Find highest priority target within 12 range that needs energy
    target_unit = None
    for unit_type in priority_unit_types:
        candidates = []
        # Shield Battery is in structures collection, others are in units
        collection = bot.structures if unit_type == UnitTypeId.SHIELDBATTERY else bot.units
        for unit in collection:
            if (unit.type_id == unit_type and
                unit.distance_to(closest_nexus) <= 12 and
                unit.energy_percentage < 1.0):
                candidates.append(unit)

        if candidates:
            target_unit = min(candidates, key=lambda u: u.energy_percentage)
            break

    if not target_unit:
        return False

    closest_nexus(AbilityId.ENERGYRECHARGE_ENERGYRECHARGE, target_unit)
    return True


def use_chronoboost(bot) -> bool:
    """
    Uses Chronoboost on the highest priority structure that would benefit from it.

    Prioritizes structures based on production/building speed impact:
    1. Research buildings (Forge, Cybernetics Core, Twilight Council, Robotics Bay)
    2. Production buildings (Gateways, Robotics Facilities, Stargates)
    3. Economic buildings (Nexus)

    Only boosts structures that are actively producing/researching and don't already have chronoboost.

    Args:
        bot: The bot instance

    Returns:
        bool: True if Chronoboost was used, False otherwise
    """
    # Recall Nexus requires 100 energy (50 reserve + 50 cost) to spend
    recall_nexus_tag = _get_recall_nexus_tag(bot)
    available_nexuses = [
        nexus for nexus in bot.townhalls.ready
        if nexus.type_id == UnitTypeId.NEXUS
        and nexus.energy >= (100 if nexus.tag == recall_nexus_tag else 50)
    ]

    if not available_nexuses:
        return False

    # Chronoboost priority: research > production > economy
    priority_structures = [
        # Research buildings
        UnitTypeId.FORGE,
        UnitTypeId.CYBERNETICSCORE,
        UnitTypeId.TWILIGHTCOUNCIL,
        UnitTypeId.ROBOTICSBAY,

        # Production buildings
        UnitTypeId.GATEWAY,
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.STARGATE,

        # Economic buildings
        UnitTypeId.NEXUS,
        

    ]

    for structure_type in priority_structures:
        # Only boost active structures without existing chronoboost and >30s remaining
        candidates = []
        for structure in bot.structures(structure_type).ready:
            if not structure.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and structure.is_active:
                remaining_time = get_remaining_activity_time(bot, structure)
                
                # Always boost probe production regardless of remaining time
                is_probe_production = (structure.type_id == UnitTypeId.NEXUS and 
                                     structure.orders and 
                                     any(order.ability.id == AbilityId.NEXUSTRAIN_PROBE for order in structure.orders))
                
                if remaining_time > 30.0 or is_probe_production:
                    candidates.append(structure)

        if candidates:
            target_structure = candidates[0]
            closest_nexus = min(available_nexuses,
                              key=lambda nexus: nexus.distance_to(target_structure.position))
            closest_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target_structure)
            return True

    return False


def get_remaining_activity_time(bot, structure) -> float:
    """
    Calculate the remaining time for a structure's current activity (building, researching, or producing).
    Uses actual game data for accurate time calculation.

    Args:
        bot: The bot instance
        structure: The structure to check

    Returns:
        float: Remaining time in seconds, or 0.0 if no significant activity
    """

    if structure.orders:
        for order in structure.orders:
            if hasattr(order, 'progress') and order.progress < 1.0:
                try:
                    ability_data = bot.game_data.abilities[order.ability.value]
                    total_time = ability_data.cost.time / 22.4  # Game time to seconds
                    remaining = (1.0 - order.progress) * total_time
                    return remaining
                except (KeyError, AttributeError):
                    # Fallback estimate when game data unavailable
                    return (1.0 - order.progress) * 45.0

    return 0.0


def use_mass_recall(bot, target_position: Point2, avoid_nexus_tag: int | None = None) -> bool:
    """
    Cast Nexus Mass Recall at target_position.

    Handles Nexus-side concerns: global cooldown, energy check, Nexus selection.
    Combat-side decision (devastating loss, retreat blocked) is in combat.py.

    Prefers a Nexus that is NOT the one closest to the army (safer, further from
    enemy pressure). Falls back to any Nexus with enough energy.

    Args:
        bot: The bot instance
        target_position: Where to cast the recall (typically army center)
        avoid_nexus_tag: Tag of the Nexus nearest to army — prefer not to use this one

    Returns:
        bool: True if recall was cast, False otherwise
    """
    # Global cooldown check (130s across all Nexuses)
    if bot.time < bot._mass_recall_last_cast_time + MASS_RECALL_COOLDOWN:
        if bot.debug:
            print(f"[MASS RECALL] Blocked by cooldown at {bot.time:.1f}s (last cast {bot._mass_recall_last_cast_time:.1f}s)")
        return False

    # Prefer the designated recall Nexus (guaranteed to have energy)
    designated_tag = _get_recall_nexus_tag(bot)
    recall_nexus = None
    if designated_tag is not None:
        for nexus in bot.structures(UnitTypeId.NEXUS).ready:
            if nexus.tag == designated_tag and nexus.energy >= MASS_RECALL_ENERGY_COST:
                recall_nexus = nexus
                break

    # Fallback: any Nexus with enough energy (prefer not the one closest to army)
    if recall_nexus is None:
        for nexus in bot.structures(UnitTypeId.NEXUS).ready:
            if nexus.energy < MASS_RECALL_ENERGY_COST:
                continue
            if avoid_nexus_tag is not None and nexus.tag != avoid_nexus_tag:
                recall_nexus = nexus
                break
        # Last resort: use the avoided Nexus
        if recall_nexus is None:
            for nexus in bot.structures(UnitTypeId.NEXUS).ready:
                if nexus.energy >= MASS_RECALL_ENERGY_COST:
                    recall_nexus = nexus
                    break

    if recall_nexus is None:
        if bot.debug:
            nexus_info = [(n.tag, f"{n.energy:.0f}e") for n in bot.structures(UnitTypeId.NEXUS).ready]
            print(f"[MASS RECALL] No nexus with {MASS_RECALL_ENERGY_COST} energy | Nexuses: {nexus_info}")
        return False

    if bot.debug:
        print(f"[MASS RECALL] Casting from nexus {recall_nexus.tag} (energy={recall_nexus.energy:.0f}) at {bot.time:.1f}s -> {target_position.round(1)}")
    recall_nexus(AbilityId.EFFECT_MASSRECALL_NEXUS, target_position)
    bot._mass_recall_last_cast_time = bot.time

    if bot.debug:
        print(f"[MASS RECALL] Cast at {bot.time:.1f}s | Nexus: {recall_nexus.tag} | Target: {target_position.round(1)}")

    return True
