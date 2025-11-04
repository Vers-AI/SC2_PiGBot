# bot/managers/structure_manager.py

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.units import Units


def use_recharge(bot, main_army: Units) -> bool:
    """
    Uses Energy Recharge on the unit with the lowest energy percentage within range of the closest Nexus.

    Targets specific energy-using units: Mothership, High Templar, Oracle, and Sentry.

    Args:
        bot: The bot instance
        main_army: Units to consider for the army center calculation

    Returns:
        bool: True if Energy Recharge was used, False otherwise
    """
    #TODO Test this
    if not main_army:
        return False

    # Define energy-using unit types in priority order
    priority_unit_types = [
        UnitTypeId.MOTHERSHIP,
        UnitTypeId.HIGHTEMPLAR,
        UnitTypeId.ORACLE,
        UnitTypeId.SENTRY
    ]

    # Find the closest Nexus to the army center
    closest_nexus = None
    closest_distance = float("inf")
    for nexus in bot.structures(UnitTypeId.NEXUS).ready:
        if nexus.energy < 50:  # Need enough energy for the ability
            continue
        distance = main_army.center.distance_to(nexus.position)
        if distance < closest_distance:
            closest_distance = distance
            closest_nexus = nexus

    if not closest_nexus:
        return False

    # Find target unit by priority order
    target_unit = None
    for unit_type in priority_unit_types:
        # Get all units of this type within range that need energy
        candidates = []
        for unit in bot.units:
            if (unit.type_id == unit_type and
                unit.distance_to(closest_nexus) <= 12 and
                unit.energy_percentage < 1.0):  # Unit needs energy
                candidates.append(unit)

        # If we found candidates of this priority type, pick the one with lowest energy
        if candidates:
            target_unit = min(candidates, key=lambda u: u.energy_percentage)
            break

    if not target_unit:
        return False

    # Cast Energy Recharge on the target unit
    closest_nexus(AbilityId.ENERGYRECHARGE_ENERGYRECHARGE, target_unit)
    return True


def use_chronoboost(bot) -> bool:
    """
    Uses Chronoboost on the highest priority structure that would benefit from it.

    Prioritizes structures based on production/building speed impact:
    1. Production buildings (Gateways, Robotics Facilities, Stargates)
    2. Research buildings (Cybernetics Core, Twilight Council, Robotics Bay, Forge)
    3. Economic buildings (last priority - Nexus, Assimilators)

    Only boosts structures that are actively producing/researching and don't already have chronoboost.

    Args:
        bot: The bot instance

    Returns:
        bool: True if Chronoboost was used, False otherwise
    """
    # Find available nexus with sufficient energy
    available_nexuses = [
        nexus for nexus in bot.townhalls.ready
        if nexus.energy >= 50 and nexus.type_id == UnitTypeId.NEXUS
    ]

    if not available_nexuses:
        return False

    # Priority order for chronoboosting (production > research > economy)
    priority_structures = [
        # Research buildings - highest priority
        UnitTypeId.FORGE,
        UnitTypeId.CYBERNETICSCORE,
        UnitTypeId.TWILIGHTCOUNCIL,
        UnitTypeId.ROBOTICSBAY,

        # Production buildings - medium priority
        UnitTypeId.GATEWAY,
        UnitTypeId.ROBOTICSFACILITY,
        UnitTypeId.STARGATE,

        # Economic buildings - lowest priority
        UnitTypeId.NEXUS,
        

    ]

    # Find the best structure to chronoboost
    for structure_type in priority_structures:
        # Get structures of this type that are ready, not already chronoboosted, actively working,
        # and have more than 30 seconds remaining in their current activity
        candidates = []
        for structure in bot.structures(structure_type).ready:
            if not structure.has_buff(BuffId.CHRONOBOOSTENERGYCOST) and structure.is_active:
                # Check if there's more than 30 seconds remaining
                remaining_time = get_remaining_activity_time(bot, structure)
                
                # Probes are an exception - always chronoboost them even if < 30 seconds
                is_probe_production = (structure.type_id == UnitTypeId.NEXUS and 
                                     structure.orders and 
                                     any(order.ability.id == AbilityId.NEXUSTRAIN_PROBE for order in structure.orders))
                
                if remaining_time > 30.0 or is_probe_production:
                    candidates.append(structure)

        if candidates:
            # Find the closest nexus to the first candidate
            target_structure = candidates[0]
            closest_nexus = min(available_nexuses,
                              key=lambda nexus: nexus.distance_to(target_structure.position))

            # Cast chronoboost
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

    # Check for production or research orders
    if structure.orders:
        for order in structure.orders:
            # Production orders have progress (0.0 to 1.0)
            if hasattr(order, 'progress') and order.progress < 1.0:
                try:
                    # Get actual time cost from game data
                    ability_data = bot.game_data.abilities[order.ability.value]
                    total_time = ability_data.cost.time / 22.4  # Convert to seconds
                    remaining = (1.0 - order.progress) * total_time
                    return remaining
                except (KeyError, AttributeError):
                    # Fallback if we can't get the data
                    return (1.0 - order.progress) * 45.0

    # If structure is idle or we can't determine, don't chronoboost
    return 0.0
