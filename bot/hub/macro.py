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
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
)
from ares.dicts.unit_data import UNIT_DATA

from bot.hub.scouting import control_scout
from bot.hub.combat import COMMON_UNIT_IGNORE_TYPES

# Army composition constants 
STANDARD_ARMY = {
    UnitTypeId.IMMORTAL: {"proportion": 0.2, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.1, "priority": 4},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.35, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 3},
    UnitTypeId.ZEALOT: {"proportion": 0.25, "priority": 5},
}

CHEESE_DEFENSE_ARMY = {
    UnitTypeId.ZEALOT: {"proportion": 0.5, "priority": 0},
    UnitTypeId.STALKER: {"proportion": 0.4, "priority": 1},
    UnitTypeId.ADEPT: {"proportion": 0.1, "priority": 2},
}


def calculate_optimal_worker_count(bot) -> int:
    """
    Calculates the optimal number of workers based on available resources.
    Checks mineral contents and vespene contents to determine if mining positions are viable.
    
    Returns:
        int: The optimal number of workers to build
    """
    mineral_spots = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        for mineral in nearby_minerals:
            # Only count if mineral field has substantial minerals left
            if mineral.mineral_contents > 100:  
                mineral_spots += 2  # 2 workers per mineral field
    
    gas_spots = 0
    for gas_building in bot.gas_buildings:
        if gas_building.vespene_contents > 100:
            gas_spots += 3  # 3 workers per gas
    
    # Add a small buffer for worker replacements
    worker_count = mineral_spots + gas_spots + 4
    
    # Debug info
    # print(f"Optimal worker count: {worker_count} (Mineral spots: {mineral_spots}, Gas spots: {gas_spots})")
    
    return worker_count


def expansion_checker(bot, main_army) -> int:
    """
    Evaluates multiple factors to determine when to expand:
    1. Game state (early, mid, late)
    2. Mineral collection rate
    3. Army value comparison between player and enemy
    4. Current base count and worker saturation
    
    Returns the recommended expansion count.
    """
    # Get current state
    current_bases = len(bot.townhalls)
    current_workers = bot.workers.amount
    optimal_workers = current_bases * 22  # 16 for minerals + 6 for gas
    worker_saturation = current_workers / optimal_workers if optimal_workers > 0 else 0
    
    # Get collection rates
    mineral_collection_rate = bot.state.score.collection_rate_minerals
    vespene_collection_rate = bot.state.score.collection_rate_vespene
    
    # Get our army value
    own_army_value = 0
    for unit in main_army:
        if unit.type_id in UNIT_DATA:
            own_army_value += UNIT_DATA[unit.type_id]['army_value']
    
    # Get enemy army value by iterating through all enemy units
    # Excluding units from COMMON_UNIT_IGNORE_TYPES
    enemy_army_value = 0
    for unit in bot.mediator.get_all_enemy:
        if unit.type_id in UNIT_DATA and unit.type_id not in COMMON_UNIT_IGNORE_TYPES:
            enemy_army_value += UNIT_DATA[unit.type_id]['army_value']
    
    # Set collection rate threshold based on game state
    if bot.game_state == "early":
        collection_threshold = 300
    elif bot.game_state == "mid":
        collection_threshold = 400
    else:  # late game
        collection_threshold = 500
    
    # Default to current number of bases
    expansion_count = current_bases
    
    # Debug information
    # print(f"Game state: {bot.game_state}")
    # print(f"Worker saturation: {worker_saturation:.2f} ({current_workers}/{optimal_workers})")
    # print(f"Mineral collection rate: {mineral_collection_rate}")
    # print(f"Own army value: {own_army_value}")
    # print(f"Enemy army value: {enemy_army_value}")
    
    # Step 1: Check if collection rate is below threshold
    if mineral_collection_rate < collection_threshold:
        # Step 2: Check if we have saturation 
        if worker_saturation > 0.8:
            # Step 3: Check army values
            if own_army_value > enemy_army_value * 0.8 or own_army_value > 1000:
                # Safe to expand
                expansion_count = current_bases + 1
                # print(f"Expanding based on army advantage: {expansion_count}")
    else:
        # If we have good income, check if we can expand based on worker saturation
        if worker_saturation > 0.8:
            expansion_count = current_bases + 1
            # print(f"Expanding based on high income and saturation: {expansion_count}")
    
    # Timing-based fallback expansions
    if bot.time > 5 * 60 and current_bases < 2:
        expansion_count = max(expansion_count, 2)
        # print(f"Fallback expansion to 2 bases at 5 min: {expansion_count}")
    elif bot.time > 8 * 60 and current_bases < 3:
        expansion_count = max(expansion_count, 3)
        # print(f"Fallback expansion to 3 bases at 8 min: {expansion_count}")
    
    # Safety check - don't expand too early with little army
    if current_bases == 1 and len(main_army) < 5 and bot.game_state == "early":
        expansion_count = 1
        # print("Limiting expansion count due to safety concerns")
    
    return expansion_count


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
    
    if not bot._used_cheese_response:
        print("Standard macro")
        macro_plan: MacroPlan = MacroPlan()
        macro_plan.add(AutoSupply(base_location=bot.start_location))
        macro_plan.add(ProductionController(STANDARD_ARMY, base_location=bot.start_location, should_repower_structures=True))
        
        # Calculate optimal worker count based on available resources
        optimal_worker_count = calculate_optimal_worker_count(bot)
        bot.register_behavior(BuildWorkers(to_count=optimal_worker_count))
        
        bot.register_behavior(
                GasBuildingController(to_count=len(bot.townhalls)*2, max_pending=2)
            )
        
        # Use the expansion_checker to determine the expansion count
        expansion_count = expansion_checker(bot, main_army)
        
        # Register the expansion controller with the current expansion_count
        bot.register_behavior(
            ExpansionController(to_count=expansion_count, max_pending=1)
        )
                
        # Spawn units near Warp Prism if available, else at base
        if warp_prism:
            prism_position = warp_prism[0].position
            macro_plan.add(
                SpawnController(STANDARD_ARMY, spawn_target=prism_position, freeflow_mode=False)
            )
        else:
            macro_plan.add(SpawnController(STANDARD_ARMY, freeflow_mode=False))

        bot.register_behavior(macro_plan)

    # If we detected cheese
    else:
        print("Cheese reaction")
        # Calculate optimal worker count for cheese defense too
        optimal_worker_count = calculate_optimal_worker_count(bot)
        bot.register_behavior(BuildWorkers(to_count=optimal_worker_count))
        
        if bot.game_state == "early":
            bot.register_behavior(
                ExpansionController(to_count=3, max_pending=1)
            )
            bot.register_behavior(
                GasBuildingController(to_count=len(bot.townhalls)*2, max_pending=2)
            )

            # Build a cheese defense plan
            cheese_defense_plan: MacroPlan = MacroPlan()
            cheese_defense_plan.add(AutoSupply(base_location=bot.start_location))
            cheese_defense_plan.add(
                SpawnController(CHEESE_DEFENSE_ARMY, spawn_target=bot.start_location, freeflow_mode=freeflow)
            )
            cheese_defense_plan.add(
                ProductionController(CHEESE_DEFENSE_ARMY, base_location=bot.start_location, should_repower_structures=True)
            )

            bot.register_behavior(cheese_defense_plan)
        else:
            #switch to the mid game build order
            bot._used_cheese_response = False
            bot._cheese_reaction_completed = True
            print("Cheese reaction completed")
        


    # Scout control or build observer if no scout
    if scout_units and main_army:
        control_scout(bot, scout_units, main_army)
    else:
        if bot.game_state == "mid":
            if bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
                if (bot.units(UnitTypeId.OBSERVER).amount < 1 
                    and bot.already_pending(UnitTypeId.OBSERVER) == 0
                    and bot.can_afford(UnitTypeId.OBSERVER)):
                    bot.train(UnitTypeId.OBSERVER)

    

    # Merge Archons if we have at least 2 High Templars
    if bot.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
        for templar in bot.units(UnitTypeId.HIGHTEMPLAR).ready:
            templar(AbilityId.MORPH_ARCHON)
