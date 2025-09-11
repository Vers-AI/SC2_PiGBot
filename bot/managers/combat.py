import numpy as np

from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from ares.consts import LOSS_MARGINAL_OR_WORSE, TIE_OR_BETTER, UnitTreeQueryType, EngagementResult, VICTORY_DECISIVE_OR_BETTER

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, PathUnitToTarget, StutterUnitBack, AMove
)
from ares.behaviors.combat.group import (
    AMoveGroup, PathGroupToTarget
)
from ares.managers.squad_manager import UnitSquad

from ares.consts import UnitRole
from ares.dicts.unit_data import UNIT_DATA

from bot.utilities.use_disruptor_nova import UseDisruptorNova
from bot.utilities.nova_manager import NovaManager

from cython_extensions import (
   
    cy_find_units_center_mass, cy_pick_enemy_target, cy_closest_to, cy_distance_to, cy_center, cy_distance_to_squared
    
)


#Units to ignore
COMMON_UNIT_IGNORE_TYPES: set[UnitTypeId] = {
    UnitTypeId.EGG,
    UnitTypeId.LARVA,
    UnitTypeId.CREEPTUMORBURROWED,
    UnitTypeId.CREEPTUMORQUEEN,
    UnitTypeId.CREEPTUMOR,
    UnitTypeId.MULE,
    UnitTypeId.PROBE,
    UnitTypeId.SCV,
    UnitTypeId.DRONE,
    UnitTypeId.OVERLORD,
    UnitTypeId.OVERSEER,
    UnitTypeId.LOCUSTMP,
    UnitTypeId.LOCUSTMPFLYING,
    UnitTypeId.ADEPTPHASESHIFT,
    UnitTypeId.CHANGELING,
    UnitTypeId.CHANGELINGMARINE,
    UnitTypeId.CHANGELINGZEALOT,
    UnitTypeId.CHANGELINGZERGLING,
    UnitTypeId.BROODLING
}


def control_main_army(bot, main_army: Units, target: Point2, squads: list[UnitSquad]) -> Point2:
    """
    Controls the main army's movement and engagement logic.
    """
    pos_of_main_squad: Point2 = bot.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
    grid: np.ndarray = bot.mediator.get_ground_grid
    avoid_grid: np.ndarray = bot.mediator.get_ground_avoidance_grid
    
    if main_army:
        bot.total_health_shield_percentage = (
            sum(unit.shield_health_percentage for unit in main_army) / len(main_army)
        )

    for squad in squads:
        maneuver = CombatManeuver()
        squad_position: Point2 = squad.squad_position
        units: list[Unit] = squad.squad_units
        squad_tags = squad.tags

        # Find nearby enemy units (excluding unimportant ones)
        all_close = bot.mediator.get_units_in_range(
            start_points=[squad_position],
            distances=13,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )[0].filter(lambda u: not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES)

        # Basic engagement logic
        if all_close:
            # Separate melee, ranged and spell casters
            melee = [u for u in units if u.ground_range <= 3 and (u.energy == 0 and u.can_attack)]
            ranged = [u for u in units if u.ground_range > 3 and (u.energy == 0 and u.can_attack)]
            spellcasters = [u for u in units if u.energy > 0 or not u.can_attack]
            # Simple picking logic
            enemy_target = cy_pick_enemy_target(all_close)
            # Ranged micro example
            for r_unit in ranged:
                ranged_maneuver = CombatManeuver()
                closest_enemy = cy_closest_to(r_unit.position, all_close)
                if not r_unit.weapon_ready:
                    ranged_maneuver.add(KeepUnitSafe(r_unit, avoid_grid))
                    ranged_maneuver.add(StutterUnitBack(r_unit, target=closest_enemy, grid=grid))
                else:
                    ranged_maneuver.add(AMove(r_unit, target=closest_enemy.position))
                bot.register_behavior(ranged_maneuver)

            # Melee engages directly
            if melee:
                melee_maneuver = CombatManeuver()
                melee_maneuver.add(AMoveGroup(group=melee, group_tags={u.tag for u in melee}, target=enemy_target.position))
                bot.register_behavior(melee_maneuver)
            
            if spellcasters:
                disruptors= [spellcaster for spellcaster in spellcasters if spellcaster.type_id == UnitTypeId.DISRUPTOR]
                for disruptor in disruptors:
                    # Execute the Nova ability if it's available, using the Nova manager for coordination
                    
                    if AbilityId.EFFECT_PURIFICATIONNOVA in disruptor.abilities:
                        try:
                            nova_manager = bot.nova_manager if hasattr(bot, 'nova_manager') else None
                            result = bot.use_disruptor_nova.execute(disruptor, all_close, units, nova_manager)
                        except Exception as e:
                            print(f"DEBUG ERROR in disruptor handling: {e}")
                    else:
                        bot.register_behavior(KeepUnitSafe(disruptor, avoid_grid))
                # Update Nova Manager with current units
                if hasattr(bot, 'nova_manager'):
                    try:
                        # Get all visible enemy units within a reasonable search range
                        enemy_units = all_close
                        friendly_units = units
                        
                        # Update the Nova Manager with current units
                        bot.nova_manager.update_units(enemy_units, friendly_units)
                        # Run the update method to handle active Novas
                        bot.nova_manager.update(enemy_units, friendly_units)
                    except Exception as e:
                        print(f"DEBUG ERROR updating NovaManager: {e}")
                    
        else:
            # No enemies nearby—regroup or move to final target
            dist_squad_main = pos_of_main_squad.distance_to(squad_position)
            if dist_squad_main > 0.1:
                maneuver.add(
                    PathGroupToTarget(
                        start=squad_position,
                        group=units,
                        group_tags=squad_tags,
                        target=pos_of_main_squad,
                        grid=avoid_grid,
                        sense_danger=False,
                        success_at_distance=0.1
                    )
                )
            else:
                maneuver.add(AMoveGroup(group=units, group_tags=squad_tags, target=target.position))

            bot.register_behavior(maneuver)

    return pos_of_main_squad

def gatekeeper_control(bot, gatekeeper: Units) -> None:
    gate_keep_pos = bot.gatekeeping_pos
    any_close = bot.mediator.get_units_in_range(
        start_points=[gate_keep_pos],
        distances=6,
        query_tree=UnitTreeQueryType.EnemyGround,
        return_as_dict=False,
    )[0]
    

    
    if not gate_keep_pos:
        return
    for gate in gatekeeper:
        dist_to_gate: float = cy_distance_to_squared(gate.position, gate_keep_pos)
        if any_close:
            if dist_to_gate > 1.0:
                gate.move(gate_keep_pos)
                gate(AbilityId.HOLDPOSITION, queue=True)
            else:
                gate(AbilityId.HOLDPOSITION)
        else:
            # Move 1 distance towards the natural expansion
            direction = bot.natural_expansion - gate_keep_pos
            if direction.length > 0:  # Avoid division by zero
                # Normalize to get direction vector with length 1
                direction = direction.normalized
                # Calculate position 1 unit from gate_keep_pos towards natural
                target_pos = gate_keep_pos + (direction * 3.0)
                gate.move(target_pos)   
            

def warp_prism_follower(bot, warp_prisms: Units, main_army: Units) -> None:
    """
    Controls Warp Prisms: follows army, morphs between Transport/Phasing.
    """
    air_grid: np.ndarray = bot.mediator.get_air_grid
    if not warp_prisms:
        return

    maneuver: CombatManeuver = CombatManeuver()
    for prism in warp_prisms:
        if main_army:
            distance_to_center = prism.distance_to(main_army.center)

            # If close, morph to Phasing
            if distance_to_center < 15:
                prism(AbilityId.MORPH_WARPPRISMPHASINGMODE)
            else:
                # If no warpin in progress, revert to Transport
                # Or simply path near the army
                not_ready_units = [unit for unit in bot.units if not unit.is_ready and unit.distance_to(prism) < 6.5]
                if prism.type_id == UnitTypeId.WARPPRISMPHASING and not not_ready_units:
                        prism(AbilityId.MORPH_WARPPRISMTRANSPORTMODE)

                elif prism.type_id == UnitTypeId.WARPPRISM:
                    # Keep prism ~3 distance behind the army center
                    direction_vector = (prism.position - main_army.center).normalized
                    new_target = main_army.center + direction_vector * 3
                    maneuver.add(PathUnitToTarget(unit=prism, target=new_target, grid=air_grid, danger_distance=10))
        else:
            # If no main army, just retreat to natural (or wherever)
            maneuver.add(
                PathUnitToTarget(
                    unit=prism,
                    target=bot.natural_expansion,
                    grid=air_grid,
                    danger_distance=10
                )
            )

    bot.register_behavior(maneuver)

#TODO Move all threat logics to reactions.py
def assess_threat(bot, enemy_units: Units, own_forces: Units) -> int: 
    """
    Assigns a 'threat level' based on unit composition.
    You can refine logic to suit your needs.
    """
    threat_level = 0
    
    for unit in enemy_units:
        weight = UNIT_DATA[unit.type_id]['army_value']
        # Multiply weight by the effective power (health + shield) scaled by 50.0, tweak if needed
        threat_level += weight * ((unit.health + unit.shield) / 50.0)

    # Density check: adjust threat level based on enemy clustering
    if enemy_units.amount > 0:
        center = cy_center(enemy_units)

        cluster_count = 0
        for unit in enemy_units:
            if cy_distance_to(unit.position, center) <= 5.0:
                cluster_count += 1

        # If fewer than 3 enemy units are clustered, scale down the threat level
        if cluster_count < 3:
            threat_level *= 0.5

    # Adjust threat if our forces significantly outnumber enemy units (not used, caused distortion)
    # if own_forces.amount > enemy_units.amount * 2:
    #     threat_level -= 2

    return max(round(threat_level), 0)


def threat_detection(bot, main_army: Units) -> None:
    """
    Detects if we have ground or air enemies near bases; calls 
    assess_threat and possibly redirects the main army to defend.
    """
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases

    if ground_near or flying_near:
        # Combine ground + air threats
        all_threats = {}
        for key, value in ground_near.items():
            all_threats[key] = value.copy()
        for key, value in flying_near.items():
            all_threats.setdefault(key, set()).update(value)

        for _, enemy_tags in all_threats.items():
            enemy_units = bot.enemy_units.tags_in(enemy_tags)
            # Begin two-threshold (hysteresis) defense logic:
            threat = assess_threat(bot, enemy_units, main_army)
            
            if bot.time < 3 * 60 and bot._used_cheese_response:
                high_threshold = 2
                low_threshold = 1
            else:
                high_threshold = 5
                low_threshold = 2
            
            if bot._under_attack:
                if threat < low_threshold:
                    bot._under_attack = False
            else:
                if threat >= high_threshold:
                    bot._under_attack = True
            
            if bot._under_attack:
                threat_position, num_units = cy_find_units_center_mass(enemy_units, 10.0)
                threat_position = Point2(threat_position)
                control_main_army(bot, main_army, threat_position, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
               


def handle_attack_toggles(bot, main_army: Units, attack_target: Point2) -> None:
    """
    Handles the attack toggles logic.
    
    Controls when to attack and when to retreat based on relative army strength
    and threat assessment.
    """
    
    # Assess the threat level of enemy units
    enemy_threat_level = assess_threat(bot, bot.enemy_units, main_army)

    enemy_army = bot.enemy_army
    # Early game safety - don't attack during cheese or one-base reactions
    is_early_defensive_mode = bot._used_cheese_response or bot._used_one_base_response
    # Only clear early defensive mode when the one-base reaction is completed
    if (is_early_defensive_mode and bot._one_base_reaction_completed) or bot.game_state == 1:  # mid game
        is_early_defensive_mode = False
    
    # Debug info
    if bot.debug:
        fight_result = bot.mediator.can_win_fight(own_units=bot.own_army, enemy_units=bot.enemy_army, timing_adjust=True, good_positioning=True, workers_do_no_damage=True)
        print(f"Can win fight: {fight_result}")
        print(f"Enemy army: {bot.enemy_army}")
        print(f"Own army: {bot.own_army}")

        bot.client.debug_text_2d(f"Attack: {bot._commenced_attack} Threat: {enemy_threat_level} Under Attack: {bot._under_attack}", 
                                Point2((0.1, 0.22)), None, 14)
        bot.client.debug_text_2d(f"EarlyDefMode: {is_early_defensive_mode} Cheese: {bot._used_cheese_response} OneBase: {bot._used_one_base_response}", 
                                Point2((0.1, 0.24)), None, 14)

    # If the army is already attacking, decide whether to continue or retreat
    if bot._commenced_attack:
        # Immediately retreat if we enter an early defensive mode
        if is_early_defensive_mode:
            bot._commenced_attack = False
            nearest_base = bot.townhalls.closest_to(main_army.center)
            if nearest_base:
                control_main_army(bot, main_army, nearest_base.position, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
        # Decide whether to retreat based on army ratio
        else:
            fight_result = bot.mediator.can_win_fight(own_units=bot.own_army, enemy_units=bot.enemy_army, timing_adjust=True, good_positioning=True, workers_do_no_damage=True)
            if fight_result in LOSS_MARGINAL_OR_WORSE:
                # Retreat to the closest base if we're outmatched
                bot._commenced_attack = False
                nearest_base = bot.townhalls.closest_to(main_army.center)
                if nearest_base:
                    control_main_army(bot, main_army, nearest_base.position, 
                                    bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
            # Only override target if threat is very high (consolidated logic)
            elif enemy_threat_level >= 5:
                enemy_center, _ = cy_find_units_center_mass(bot.enemy_units, 10.0)
                control_main_army(bot, main_army, Point2(enemy_center), 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
            # Otherwise, stick with current attack target for stability
            else:
                control_main_army(bot, main_army, attack_target, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
    else:
        # Check our fighting capability first
        fight_result = bot.mediator.can_win_fight(own_units=bot.own_army, enemy_units=bot.enemy_army, timing_adjust=True, good_positioning=True, workers_do_no_damage=True)
        
        # Attack if we have a clear advantage (decisive victory or better), regardless of defensive flags
        if fight_result in VICTORY_DECISIVE_OR_BETTER:
            control_main_army(bot, main_army, attack_target, 
                            bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
            bot._commenced_attack = True
        # Only consider attacking if build order is complete and not in early defensive mode
        elif not is_early_defensive_mode:
            if (fight_result in TIE_OR_BETTER and not bot._under_attack):
                control_main_army(bot, main_army, attack_target, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
                bot._commenced_attack = True
        # When build order isn't complete or in defensive mode, focus on defending if needed
        elif bot._under_attack and bot.enemy_units:
            enemy_center, _ = cy_find_units_center_mass(bot.enemy_units, 10.0)
            control_main_army(bot, main_army, Point2(enemy_center), 
                            bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))


def attack_target(bot, main_army_position: Point2) -> Point2:
    """
    Determines where our main attacking units should move toward.
    Enhanced with target persistence to reduce bouncing behavior.
    """

    # Check if we should prioritize enemy army over structures (borrowed from example)
    enemy_army = bot.enemy_units.filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES and not u.is_structure)
    enemy_supply = sum(bot.calculate_supply_cost(u.type_id) for u in enemy_army)
    
    if enemy_supply >= 6 and enemy_army:
        enemy_center, _ = cy_find_units_center_mass(enemy_army, 10.0)
        # If we have significant enemy forces, prioritize them over structures
        all_close_enemy = bot.mediator.get_units_in_range(
            start_points=[Point2(enemy_center)],
            distances=11.5,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=False,
        )[0]
        if sum(bot.calculate_supply_cost(u.type_id) for u in all_close_enemy) >= 7:
            bot.current_attack_target = Point2(enemy_center)
            return bot.current_attack_target

    if bot.enemy_structures:
        closest_structure = cy_closest_to(main_army_position, bot.enemy_structures)
        
        # Distance-based stickiness - if we're close to enemy structures, don't get distracted
        if (bot.current_attack_target is not None and 
            cy_distance_to_squared(main_army_position, bot.current_attack_target) < 450.0):
            return bot.current_attack_target
        
        # If we have a current target, only switch if new target is significantly closer
        if bot.current_attack_target is not None:
            current_distance = main_army_position.distance_to(bot.current_attack_target)
            new_distance = main_army_position.distance_to(closest_structure.position)
            
            # Only switch if new target is meaningfully closer
            if new_distance < current_distance - bot.target_lock_distance:
                bot.current_attack_target = Point2(closest_structure.position)
                
            # Check if current target is still valid (structure still exists nearby)
            elif (bot.is_visible(bot.current_attack_target) and 
                  not bot.enemy_structures.closer_than(5, bot.current_attack_target)):
                # Target area is clear, allow new target selection
                bot.current_attack_target = Point2(closest_structure.position)
        else:
            # First time setting target
            bot.current_attack_target = Point2(closest_structure.position)
            
        return bot.current_attack_target
        
    # Not seen anything in early game, just head to enemy spawn
    elif bot.time < 240.0:
        fallback = bot.enemy_start_locations[0]
        bot.current_attack_target = fallback
        return fallback
    
    # Else search the map
    else:
        fallback = fallback_target(bot)
        bot.current_attack_target = fallback
        return fallback


def fallback_target(bot) -> Point2:
    """
    Provides a fallback target for the main army when no clear target is available.
    Cycles through expansion locations with mineral field validation (borrowed from example).
    """
    # Cycle through expansion locations with better validation
    if bot.is_visible(bot.current_base_target):
        # Check if there are mineral fields near this base location
        if mineral_fields := [
            mf for mf in bot.mineral_field 
            if cy_distance_to_squared(mf.position, bot.current_base_target) < 144.0
        ]:
            closest_mineral = cy_closest_to(bot.current_base_target, mineral_fields).position
            if bot.is_visible(closest_mineral):
                bot.current_base_target = next(bot.expansions_generator)
        else:
            bot.current_base_target = next(bot.expansions_generator)
    
    return bot.current_base_target

#TODO check regrouping too aggressive? 
def regroup_army(bot, main_army: Units) -> None:
    """Regroups the main army if units are too scattered and the bot is not in a combat state.

    Conditions:
      - The average distance of units from the center of mass is greater than 10 units.
      - The bot is not currently under attack and has not commenced an attack.
    Action:
      - Command the main army to move to the natural expansion location (bot.natural_expansion).
    """
    if not main_army or main_army.amount == 0:
        return

    # Calculate the center of mass of the main army
    center, _ = cy_find_units_center_mass(main_army, 10.0)
    center = Point2(center)

    # Compute the average distance from the center
    total_distance = 0.0
    for unit in main_army:
        total_distance += unit.position.distance_to(center)
    avg_distance = total_distance / main_army.amount

    # If the army is scattered (avg distance > 15) and not in combat, regroup
    if avg_distance > 15 and not bot._under_attack and not bot._commenced_attack:
        # Command the main army to regroup by using control_main_army with the natural expansion as target
        control_main_army(bot, main_army, bot.natural_expansion.towards(bot.game_info.map_center, 2), bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
        print("Regrouping army")


