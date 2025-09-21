# bot/managers/macro.py

from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId

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
    UpgradeController,
)
from ares.dicts.unit_data import UNIT_DATA
from ares.consts import LOSS_MARGINAL_OR_BETTER

from bot.managers.scouting import control_scout
from bot.managers.combat import COMMON_UNIT_IGNORE_TYPES

# Army composition constants 
STANDARD_ARMY_0 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.2, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.1, "priority": 4},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.35, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 3},
    UnitTypeId.ZEALOT: {"proportion": 0.25, "priority": 5},
}
STANDARD_ARMY_1 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.35, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.2, "priority": 4},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 3},
    UnitTypeId.ZEALOT: {"proportion": 0.35, "priority": 0},
}

PVP_ARMY_0 = {
    UnitTypeId.DISRUPTOR: {"proportion": 0.2, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 3},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.35, "priority": 0},
    UnitTypeId.IMMORTAL: {"proportion": 0.3, "priority": 2},
}

PVP_ARMY_1 = {
    UnitTypeId.DISRUPTOR: {"proportion": 0.25, "priority": 0},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 2},
    UnitTypeId.IMMORTAL: {"proportion": 0.6, "priority": 1},
}

CHEESE_DEFENSE_ARMY = {
    UnitTypeId.ZEALOT: {"proportion": 0.5, "priority": 0},
    UnitTypeId.STALKER: {"proportion": 0.4, "priority": 1},
    UnitTypeId.ADEPT: {"proportion": 0.1, "priority": 2},
}

#Upgrades
desired_upgrades: list[UpgradeId] = [
    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
    UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
    UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
    UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
]

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
    
    
    # Set collection rate threshold based on game state
    if bot.game_state == 0:  # early game
        collection_threshold = 300
    elif bot.game_state == 1:  # mid game
        collection_threshold = 600
    else:  # late game
        collection_threshold = 1000
    
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
            if bot.mediator.can_win_fight(
                own_units=bot.own_army,
                enemy_units=bot.enemy_army,
                timing_adjust=True,
                good_positioning=False,
                workers_do_no_damage=True,
            ) in LOSS_MARGINAL_OR_BETTER:
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
    if current_bases == 1 and len(main_army) < 5 and bot.game_state == 0:  # early game
        expansion_count = 1
        # print("Limiting expansion count due to safety concerns")
    
    return expansion_count


def select_army_composition(bot, main_army: Units) -> dict:
    """
    Determines which army composition to use based on the current army state and enemy race.
    For PVP: Uses PVP_ARMY_0/1 compositions
    For other matchups: Uses STANDARD_ARMY_0/1 compositions
    Switches between versions when Archons reach a threshold percentage.
    
    Returns:
        dict: The selected army composition dictionary
    """
    # Select composition set based on enemy race
    if bot.enemy_race == Race.Protoss:
        # PVP compositions
        selected_composition = PVP_ARMY_0
        army_1 = PVP_ARMY_1
        threshold = 0.30  # Higher threshold for PVP (30%)
        comp_name = "PVP"
    else:
        # Standard compositions for PVZ and PVT
        selected_composition = STANDARD_ARMY_0
        army_1 = STANDARD_ARMY_1
        threshold = 0.15  # Standard threshold (15%)
        comp_name = "STANDARD"
    
    # Only evaluate composition switch if we have a main army
    if main_army and len(main_army) > 0:
        # Count Archons in the main army
        archon_count = sum(1 for unit in main_army if unit.type_id == UnitTypeId.ARCHON)
        
        # Calculate the percentage of Archons in the army
        archon_percentage = archon_count / len(main_army) if len(main_army) > 0 else 0
        
        # Debug info
        if bot.debug:
            print(f"{comp_name} Archon percentage: {archon_percentage:.2f} ({archon_count}/{len(main_army)})")
        
        # Switch to version 1 when Archons reach threshold
        if archon_percentage >= threshold:
            selected_composition = army_1
            if bot.debug:
                print(f"Switching to {comp_name}_ARMY_1 due to high Archon count")
    
    return selected_composition


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
    spawn_location = bot.natural_expansion
    # spawn_location = bot.start_location #need change this to confirm if there is a pylon in the spawn location
    production_location = bot.start_location

    if not bot._used_cheese_response:
        macro_plan: MacroPlan = MacroPlan()
        
        # Select army composition based on current army state
        army_composition = select_army_composition(bot, main_army)
        
        # Calculate optimal worker count based on available resources
        # Dynamic worker limit: 66 in mid game, 80 in late game
        worker_limit = 80 if bot.game_state >= 2 else 66
        optimal_worker_count = min(calculate_optimal_worker_count(bot), worker_limit)
        
        # Use the expansion_checker to determine the expansion count
        expansion_count = expansion_checker(bot, main_army)
        
        # Macro Plan
        macro_plan.add(BuildWorkers(to_count=optimal_worker_count))
        macro_plan.add(AutoSupply(base_location=production_location))
        macro_plan.add(GasBuildingController(to_count=len(bot.townhalls)*2, max_pending=2))
        macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1))
        
        macro_plan.add(ProductionController(army_composition, base_location=production_location, should_repower_structures=True))
        macro_plan.add(UpgradeController(desired_upgrades, base_location=production_location))
        
        # Spawn units near Warp Prism if available, else at base
        if warp_prism:
            prism_position = warp_prism[0].position
            macro_plan.add(
                SpawnController(army_composition, spawn_target=prism_position, freeflow_mode=freeflow)
            )
        else:
            macro_plan.add(SpawnController(army_composition, spawn_target=spawn_location, freeflow_mode=freeflow))

        # Register the complete macro_plan once
        bot.register_behavior(macro_plan)

    # If we detected cheese
    else:
        print("Cheese reaction")
        # Calculate optimal worker count for cheese defense too
        # Dynamic worker limit: 66 in mid game, 80 in late game
        worker_limit = 80 if bot.game_state >= 2 else 66
        optimal_worker_count = min(calculate_optimal_worker_count(bot), worker_limit)
        bot.register_behavior(BuildWorkers(to_count=optimal_worker_count))
        
        if bot.game_state == 0:  # early game
            bot.register_behavior(
                ExpansionController(to_count=3, max_pending=1)
            )
            bot.register_behavior(
                GasBuildingController(to_count=len(bot.townhalls)*2, max_pending=2)
            )

            # Build a cheese defense plan
            cheese_defense_plan: MacroPlan = MacroPlan()
            cheese_defense_plan.add(AutoSupply(base_location=production_location))
            cheese_defense_plan.add(
                SpawnController(CHEESE_DEFENSE_ARMY, spawn_target=spawn_location, freeflow_mode=True)
            )
            cheese_defense_plan.add(
                ProductionController(CHEESE_DEFENSE_ARMY, base_location=production_location, should_repower_structures=True)
            )

            bot.register_behavior(cheese_defense_plan)
        else:
            #switch to the mid game build order
            bot._used_cheese_response = False
            bot._cheese_reaction_completed = True
            print("Cheese reaction completed")
        


    # Controls how many Observers we have
    if bot.game_state == 0:  # Early game
        if (bot.units(UnitTypeId.OBSERVER).amount < 1 
            and bot.already_pending(UnitTypeId.OBSERVER) == 0
            and bot.can_afford(UnitTypeId.OBSERVER)):
            for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
                facility.train(UnitTypeId.OBSERVER)
                break
    else:  # Mid game and beyond
        # High priority observer - always build one if we don't have any
        if (bot.units(UnitTypeId.OBSERVER).amount < 1 
            and bot.already_pending(UnitTypeId.OBSERVER) == 0
            and bot.can_afford(UnitTypeId.OBSERVER)):
            for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
                facility.train(UnitTypeId.OBSERVER)
                break
        # Additional observers based on opponent race
        target_count = 3 if bot.enemy_race in {Race.Zerg, Race.Terran} else 2
        if (bot.units(UnitTypeId.OBSERVER).amount < target_count
            and bot.already_pending(UnitTypeId.OBSERVER) == 0
            and bot.can_afford(UnitTypeId.OBSERVER)):
            for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                facility.train(UnitTypeId.OBSERVER)
                break

    
    

    # Merge Archons if we have at least 2 High Templars
    if bot.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
        for templar in bot.units(UnitTypeId.HIGHTEMPLAR).ready:
            templar(AbilityId.MORPH_ARCHON)
