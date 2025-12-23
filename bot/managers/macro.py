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
    BuildStructure,
)
from ares.consts import UnitRole, WORKER_TYPES
from ares.dicts.unit_data import UNIT_DATA
from ares.consts import LOSS_MARGINAL_OR_BETTER

from bot.managers.scouting import control_scout
from bot.constants import COMMON_UNIT_IGNORE_TYPES
from bot.utilities.performance_monitor import get_economy_state


def get_freeflow_mode(bot) -> bool:
    """
    Dynamic freeflow calculation based on current economy state and spending efficiency.
    Freeflow mode ignores army composition proportions and spends resources freely.
    
    Args:
        bot: The bot instance
        
    Returns:
        bool: True if should use freeflow mode
    """
    # Both resources high - spend freely
    if bot.minerals > 500 and bot.vespene > 500:
        return True
    
    # High minerals, low gas imbalance
    if bot.minerals > 800 and bot.vespene < 200:
        return True
    
    # One base with decent bank - apply pressure
    if len(bot.townhalls) == 1 and bot.minerals >= 280 and bot.vespene >= 105:
        return True
    
    # PvP special case - one base with immortal production
    if (bot.enemy_race == Race.Protoss 
        and len(bot.townhalls) <= 1 
        and bot.unit_pending(UnitTypeId.IMMORTAL)):
        return True
    
    # SQ-based trigger: low spending quotient means we're banking too much
    if hasattr(bot, 'performance_monitor') and bot.performance_monitor.should_trigger_freeflow(bot):
        return True
    
    # Normal proportional production
    return False


def get_optimal_gas_workers(bot) -> int:
    """
    Dynamically adjust gas workers based on economy state.
    Prevents gas flooding and mineral starvation.
    
    Args:
        bot: The bot instance
        
    Returns:
        int: Number of workers per gas (0-3)
    """
    # Get current gatherers (workers actually mining)
    gatherers = bot.mediator.get_units_from_role(role=UnitRole.GATHERING)
    
    # Critical mineral starvation - stop gas completely
    if (bot.minerals < 100 
        and bot.vespene > 300 
        and bot.supply_used < 84):
        return 0
    
    # Early game with few workers - stop gas temporarily
    if ((bot._used_cheese_response and len(gatherers) < 21)
        or len(gatherers) < 12):
        return 0
    
    # Vespene flooding - toggle off gas mining
    if bot._gas_worker_toggle and bot.vespene > 1500 and bot.minerals < 100:
        bot._gas_worker_toggle = False
        return 0
    
    # Vespene starved - toggle gas mining back on
    if bot.vespene < 400 or bot.minerals > 1200:
        bot._gas_worker_toggle = True
    
    # Low gas - max workers
    if bot.vespene < 100:
        return 3
    
    return 3 if bot._gas_worker_toggle else 0

# Army composition constants 
STANDARD_ARMY_0 = {
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 1},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 4},
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.4, "priority": 0},
    UnitTypeId.DISRUPTOR: {"proportion": 0.1, "priority": 3},
    UnitTypeId.ZEALOT: {"proportion": 0.2, "priority": 2},
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
    UnitTypeId.HIGHTEMPLAR: {"proportion": 0.5, "priority": 0},
    UnitTypeId.IMMORTAL: {"proportion": 0.15, "priority": 2},
}

PVP_ARMY_1 = {
    UnitTypeId.DISRUPTOR: {"proportion": 0.25, "priority": 0},
    UnitTypeId.COLOSSUS: {"proportion": 0.15, "priority": 2},
    UnitTypeId.IMMORTAL: {"proportion": 0.6, "priority": 1},
}

CHEESE_DEFENSE_ARMY = {
    UnitTypeId.ADEPT: {"proportion": 0.25, "priority": 1},
    UnitTypeId.STALKER: {"proportion": 0.15, "priority": 0},
    UnitTypeId.ZEALOT: {"proportion": 0.6, "priority": 2},
    
    
}

# No static upgrade list - use get_desired_upgrades() instead

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
            if mineral.mineral_contents > 100:  
                mineral_spots += 2
    
    gas_spots = 0
    for gas_building in bot.gas_buildings:
        if gas_building.vespene_contents > 100:
            gas_spots += 3
    
    # Add a small buffer for worker replacements
    worker_count = mineral_spots + gas_spots + 4
    
    return worker_count


def is_base_depleted(bot) -> bool:
    """Accurate base depletion using ARES data + python-sc2 mineral contents"""
    
    # Method 1: Check mineral contents directly (python-sc2)
    depleted_bases = 0
    for townhall in bot.townhalls:
        nearby_minerals = bot.mineral_field.closer_than(10, townhall)
        if nearby_minerals:
            depleted_patches = len([m for m in nearby_minerals if m.mineral_contents < 300])
            if depleted_patches / len(nearby_minerals) > 0.5:
                depleted_bases += 1
    
    # Method 2: Use ARES worker assignment data
    mineral_assignments = bot.mediator.get_mineral_patch_to_list_of_workers
    available_patches = bot.mediator.get_num_available_mineral_patches
    
    # Count oversaturated patches (3+ workers per patch)
    oversaturated_count = sum(1 for workers in mineral_assignments.values() 
                             if len(workers) >= 3)
    
    gas_workers = len(bot.mediator.get_worker_to_vespene_dict)
    mining_workers = bot.workers.amount - gas_workers
    
    return (
        depleted_bases > 0                          # Some bases have depleted patches
        or available_patches < 4                    # Very few available patches  
        or (mining_workers > 0 and available_patches == 0)  # Workers but no patches
        or oversaturated_count > 2                  # Too many oversaturated patches
    )


def expansion_checker(bot, main_army) -> int:
    """
    Evaluates multiple factors to determine when to expand:
    1. Resource starvation (production idle due to lack of income)
    2. Base depletion (worker saturation with declining income)
    3. Army safety for expansion
    4. Spending efficiency using python-sc2 score metrics
    
    Returns the recommended expansion count.
    """
    current_bases = len(bot.townhalls)
    current_workers = bot.workers.amount
    optimal_workers = calculate_optimal_worker_count(bot)
    worker_saturation = current_workers / optimal_workers if optimal_workers > 0 else 0
    
    mineral_collection_rate = bot.state.score.collection_rate_minerals
    idle_production_time = bot.state.score.idle_production_time
    idle_worker_time = bot.state.score.idle_worker_time
    
    expansion_count = current_bases
    
    # Calculate spending efficiency (SQ-style)
    current_unspent = bot.minerals
    spending_efficiency = mineral_collection_rate / (current_unspent + 1) if mineral_collection_rate > 0 else 0
    
    # Safety check - only expand if safe (filter workers from both armies)
    own_combat_units = [u for u in bot.own_army if u.type_id not in WORKER_TYPES]
    enemy_combat_units = [u for u in bot.enemy_army if u.type_id not in WORKER_TYPES]
    army_safe = bot.mediator.can_win_fight(
        own_units=own_combat_units,
        enemy_units=enemy_combat_units,
        timing_adjust=True,
        good_positioning=False,
        workers_do_no_damage=True,
    ) in LOSS_MARGINAL_OR_BETTER
    
    if not army_safe:
        return expansion_count
    
    resource_starved = (
        idle_production_time > 30.0  # Production buildings idle for 30+ seconds
        and current_unspent < 500    # Low mineral bank
        and mineral_collection_rate > 0  # But we do have some income
    )
    
    bases_depleting = is_base_depleted(bot)
    
    # Spending efficiency thresholds (dynamic by game state)
    if bot.game_state == 0:      # Early: should spend quickly
        efficiency_threshold = 1.5
    elif bot.game_state == 1:    # Mid: more complex economy
        efficiency_threshold = 1.0  
    else:                        # Late: can bank for big investments
        efficiency_threshold = 0.5
    
    # Low spending efficiency = banking too much relative to income
    inefficient_spending = spending_efficiency < efficiency_threshold
    
    if resource_starved:
        expansion_count = current_bases + 1
    elif bases_depleting:
        expansion_count = current_bases + 1
    elif inefficient_spending and worker_saturation > 0.7:
        expansion_count = current_bases + 1
    
    # Game state-based fallback expansions (minimal safety net)
    if bot.game_state >= 1 and current_bases < 2:
        expansion_count = max(expansion_count, 2)
    elif bot.game_state >= 2 and current_bases < 3:
        expansion_count = max(expansion_count, 3)
    
    # Limit early expansion with small army
    if current_bases == 1 and len(main_army) < 5 and bot.game_state == 0:
        expansion_count = 1
    
    return expansion_count


def get_desired_upgrades(bot) -> list[UpgradeId]:
    """
    Returns dynamic upgrade list based on game state and structures.
    UpgradeController will automatically build required tech structures.
    """
    upgrades: list[UpgradeId] = []
    
    if not bot.already_pending_upgrade(UpgradeId.WARPGATERESEARCH):
        upgrades.append(UpgradeId.WARPGATERESEARCH)
    
    if (bot.structures(UnitTypeId.ROBOTICSBAY) 
        and (bot.units(UnitTypeId.COLOSSUS) or bot.already_pending(UnitTypeId.COLOSSUS))):
        upgrades.append(UpgradeId.EXTENDEDTHERMALLANCE)
    
    if bot.structures(UnitTypeId.TWILIGHTCOUNCIL) or (len(bot.townhalls) >= 3 and bot.supply_used >= 54):
        upgrades.append(UpgradeId.CHARGE)
    
    # Gate early upgrades if economy not ready (use centralized economy state)
    economy_state = get_economy_state(bot)
    if economy_state in ("recovery", "reduced"):
        return upgrades
    
    # Also gate if army is too small (need units before upgrades)
    if bot.supply_army < 15:
        return upgrades
    
    if bot.structures(UnitTypeId.FORGE) and ((len(bot.townhalls) >= 3 and bot.supply_used >= 56)):
        upgrades.extend([
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
            UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
            UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
            UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
            UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
        ])
    
    return upgrades


def require_warp_prism(bot) -> bool:
    """Returns True if WarpPrism should be built. Limits to 1 total."""
    # Check both forms: transport mode and phasing mode
    total_prisms = (bot.units(UnitTypeId.WARPPRISM).amount + 
                    bot.units(UnitTypeId.WARPPRISMPHASING).amount +
                    bot.already_pending(UnitTypeId.WARPPRISM))
    return (bot.structures(UnitTypeId.ROBOTICSFACILITY).ready
            and bot.structures(UnitTypeId.TEMPLARARCHIVE).ready
            and bot.structures(UnitTypeId.ROBOTICSBAY).ready
            and bot.supply_used >= 60
            and total_prisms == 0)


def require_shield_battery(bot) -> bool:
    """
    Returns True if shield battery should be built after cheese response.
    Limits to one battery total. Tech requirements checked by BuildStructure.
    """
    total_batteries = (bot.structures(UnitTypeId.SHIELDBATTERY).amount + 
                      bot.already_pending(UnitTypeId.SHIELDBATTERY))
    
    return bot._used_cheese_response and total_batteries == 0


def get_shield_battery_base_location(bot) -> Point2:
    """
    Determines which base location to build shield battery at:
    - At natural if buildings or townhall exist there
    - At main base otherwise (uses static_defence placement)
    """
    natural_location = bot.mediator.get_own_nat
    
    natural_has_base = bot.townhalls.closer_than(10, natural_location).amount > 0
    natural_has_structures = bot.structures.closer_than(10, natural_location).amount > 0
    
    if natural_has_base or natural_has_structures:
        return natural_location
    else:
        # Use main base - BuildStructure with static_defence=True will find defensive spot
        return bot.start_location


def get_desired_gateway_count(bot) -> int:
    """
    Returns desired gateway/warpgate count based on supply and bases.
    Build order provides 3, scales to 5, then 8 for mid-game production.
    """
    if bot.supply_used >= 62 and len(bot.townhalls) >= 5:
        return 8
    elif bot.supply_used >= 54 and len(bot.townhalls) >= 3:
        return 5
    else:
        return 3


def get_desired_forge_count(bot) -> int:
    """
    Returns desired forge count based on bases and game state.
    Build 1 after first expansion, second in late game.
    """
    if len(bot.townhalls.ready) >= 4:
        return 2
    elif len(bot.townhalls.ready) >= 2:
        return 1
    else:
        return 0


def select_army_composition(bot, main_army: Units) -> dict:
    """
    Determines which army composition to use based on the current army state and enemy race.
    For PVP: Uses PVP_ARMY_0/1 compositions
    For other matchups: Uses STANDARD_ARMY_0/1 compositions
    Switches between versions when Archons reach a threshold percentage.
    
    Returns:
        dict: The selected army composition dictionary
    """
    if bot.enemy_race == Race.Protoss:
        selected_composition = PVP_ARMY_0
        army_1 = PVP_ARMY_1
        threshold = 0.30
    else:
        selected_composition = STANDARD_ARMY_0
        army_1 = STANDARD_ARMY_1
        threshold = 0.15
    
    # Switch composition when Archon percentage exceeds threshold
    if main_army and len(main_army) > 0:
        archon_count = sum(1 for unit in main_army if unit.type_id == UnitTypeId.ARCHON)
        archon_percentage = archon_count / len(main_army) if len(main_army) > 0 else 0
        
        if archon_percentage >= threshold:
            selected_composition = army_1
    
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
    spawn_location = bot.natural_expansion
    production_location = bot.start_location
    
    worker_limit = 90 if bot.game_state >= 1 else 66
    optimal_worker_count = min(calculate_optimal_worker_count(bot), worker_limit)
    
    if not bot._used_cheese_response:
        army_composition = select_army_composition(bot, main_army)
        expansion_count = expansion_checker(bot, main_army)
    else:
        army_composition = CHEESE_DEFENSE_ARMY
        expansion_count = 3
    
    if require_shield_battery(bot):
        base_location = get_shield_battery_base_location(bot)
        bot.register_behavior(
            BuildStructure(
                base_location=base_location,
                structure_id=UnitTypeId.SHIELDBATTERY,
                static_defence=True
            )
        )
    
    if require_warp_prism(bot):
        for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
            if bot.can_afford(UnitTypeId.WARPPRISM):
                facility.train(UnitTypeId.WARPPRISM)
                break
    
    economy_state = get_economy_state(bot)
    
    # Scale gateways to desired count (3→5→8) - only in moderate+ economy
    if economy_state in ("moderate", "full"):
        desired_gates = get_desired_gateway_count(bot)
        current_gates = (bot.structures(UnitTypeId.GATEWAY).amount + 
                         bot.structures(UnitTypeId.WARPGATE).amount + 
                         bot.already_pending(UnitTypeId.GATEWAY))
        
        if current_gates < desired_gates and bot.can_afford(UnitTypeId.GATEWAY):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.GATEWAY))
        
        # Scale forges to desired count (0→1→2)
        desired_forges = get_desired_forge_count(bot)
        current_forges = (bot.structures(UnitTypeId.FORGE).amount + 
                         bot.already_pending(UnitTypeId.FORGE))
        
        if current_forges < desired_forges and bot.can_afford(UnitTypeId.FORGE):
            bot.register_behavior(BuildStructure(production_location, UnitTypeId.FORGE))
    
    macro_plan: MacroPlan = MacroPlan()
    
    # Always: workers and supply (all economy states)
    macro_plan.add(BuildWorkers(to_count=optimal_worker_count))
    macro_plan.add(AutoSupply(base_location=production_location))
    
    if economy_state == "recovery":
        pass
    else:
        # Reduced+: gas buildings and spawn from existing production
        macro_plan.add(GasBuildingController(to_count=len(bot.townhalls)*2, max_pending=2))
        
        spawn_target = warp_prism[0].position if warp_prism else spawn_location
        spawn_freeflow = True if bot._used_cheese_response else freeflow
        
        # Skip spawning if saving for expansion during reduced economy (when safe)
        saving_for_expansion = (not bot._under_attack and economy_state == "reduced" and bot.minerals < 450)
        if not saving_for_expansion:
            macro_plan.add(SpawnController(army_composition, spawn_target=spawn_target, freeflow_mode=spawn_freeflow))
        
        # Expansion logic: moderate+ gets full expansion, reduced gets safety net to 2 bases
        # Skip expansions when under attack - focus resources on defense
        if not bot._under_attack:
            if economy_state in ("moderate", "full"):
                macro_plan.add(ExpansionController(to_count=expansion_count, max_pending=1))
            elif len(bot.townhalls) < 2:
                # Safety net: always allow natural expansion even in reduced economy
                macro_plan.add(ExpansionController(to_count=2, max_pending=1))
        
        if economy_state in ("moderate", "full"):
            # Moderate+: upgrades
            macro_plan.add(UpgradeController(get_desired_upgrades(bot), base_location=production_location))
            
            if economy_state == "full":
                # Full: add new production buildings
                macro_plan.add(ProductionController(army_composition, base_location=production_location, add_production_at_bank=(450,450), ignore_below_proportion=0.3))
    
    bot.register_behavior(macro_plan)
    if bot.game_state == 0:
        if (bot.units(UnitTypeId.OBSERVER).amount < 1 
            and bot.already_pending(UnitTypeId.OBSERVER) == 0
            and bot.can_afford(UnitTypeId.OBSERVER)):
            for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
                facility.train(UnitTypeId.OBSERVER)
                break
    else:
        if (bot.units(UnitTypeId.OBSERVER).amount < 1 
            and bot.already_pending(UnitTypeId.OBSERVER) == 0
            and bot.can_afford(UnitTypeId.OBSERVER)):
            for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready:
                facility.train(UnitTypeId.OBSERVER)
                break
        target_count = 3 if bot.enemy_race in {Race.Zerg, Race.Terran} else 2
        if (bot.units(UnitTypeId.OBSERVER).amount < target_count
            and bot.already_pending(UnitTypeId.OBSERVER) == 0
            and bot.can_afford(UnitTypeId.OBSERVER)):
            for facility in bot.structures(UnitTypeId.ROBOTICSFACILITY).ready.idle:
                facility.train(UnitTypeId.OBSERVER)
                break
    
    if bot.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
        for templar in bot.units(UnitTypeId.HIGHTEMPLAR).ready:
            templar(AbilityId.MORPH_ARCHON)
