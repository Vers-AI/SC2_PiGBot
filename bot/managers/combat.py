import numpy as np

from sc2.position import Point2, Point3
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from ares.consts import LOSS_MARGINAL_OR_WORSE, TIE_OR_BETTER, UnitTreeQueryType, EngagementResult, VICTORY_DECISIVE_OR_BETTER, VICTORY_MARGINAL_OR_BETTER, LOSS_OVERWHELMING_OR_WORSE

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
from bot.managers.reactions import assess_threat, allocate_defensive_forces

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
}




def control_main_army(bot, main_army: Units, target: Point2, squads: list[UnitSquad]) -> Point2:
    """
    Controls the main army's movement and engagement logic.
    """
    # ARES requires get_squads() to be called before get_position_of_main_squad()
    if not squads:
        squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0)
    
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

        # Main squad coordination: main squad goes to target, others converge on main squad
        move_to: Point2 = target if squad.main_squad else pos_of_main_squad

        # Find nearby enemy units 
        
        all_close = bot.mediator.get_units_in_range(
            start_points=[squad_position],
            distances=9,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )[0].filter(lambda u: not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES)
        
        if all_close:
            # Terrain height awareness - avoid engaging down ramps when defending

            if (not bot._commenced_attack and len(bot.townhalls) <= 1 and bot.time < 600.0 and
                bot.get_terrain_z_height(all_close.center) < bot.get_terrain_z_height(bot.start_location) and
                any(bot.get_terrain_z_height(u.position) < bot.get_terrain_z_height(bot.start_location) 
                    for u in units)):
                # Don't engage - move back to higher ground
                for unit in units:
                    if bot.get_terrain_z_height(unit.position) < bot.get_terrain_z_height(bot.start_location):
                        unit.move(bot.start_location)
                return pos_of_main_squad

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
            # No enemies nearby - use continuous cohesion approach
            maneuver.add(AMoveGroup(group=units, group_tags=squad_tags, target=move_to))    
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


def handle_attack_toggles(bot, main_army: Units, attack_target: Point2) -> None:
    """
    Enhanced attack toggles logic with smart threat response.
    
    Controls when to attack and when to retreat based on relative army strength
    and intelligent threat assessment that prevents bouncing.
    """
    
    # Check if main army is currently handling defense (set by threat_detection)
    if hasattr(bot, '_main_army_defending') and bot._main_army_defending and not bot._under_attack:
        # Main army is handling a major threat, don't interfere
        return
    
    # Assess the threat level of enemy units for main combat decisions
    enemy_threat_level = assess_threat(bot, bot.enemy_units, main_army)  # Returns simple int
    # Type assertion since return_details=False (default) returns int
    assert isinstance(enemy_threat_level, int), "assess_threat without return_details should return int"

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
        # Removed detailed army unit printing to reduce console spam

        bot.client.debug_text_2d(f"Attack: {bot._commenced_attack} Threat: {enemy_threat_level} Under Attack: {bot._under_attack}", 
                                Point2((0.1, 0.22)), None, 14)
        bot.client.debug_text_2d(f"EarlyDefMode: {is_early_defensive_mode} Cheese: {bot._used_cheese_response} OneBase: {bot._used_one_base_response}", 
                                Point2((0.1, 0.24)), None, 14)
        
        # Role and squad debug info
        attacking_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        defending_units = bot.mediator.get_units_from_role(role=UnitRole.DEFENDING) 
        base_defender_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
        
        # Squad counts
        attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0)
        defending_squads = bot.mediator.get_squads(role=UnitRole.DEFENDING, squad_radius=9.0)
        base_defender_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=6.0)
        
        bot.client.debug_text_2d(f"ROLES: ATK:{len(attacking_units)} DEF:{len(defending_units)} BASE:{len(base_defender_units)}", 
                                Point2((0.1, 0.26)), None, 14)
        bot.client.debug_text_2d(f"SQUADS: ATK:{len(attacking_squads)} DEF:{len(defending_squads)} BASE:{len(base_defender_squads)}", 
                                Point2((0.1, 0.28)), None, 14)
        
        # Visual debug markers for targeting
        if hasattr(bot, 'current_attack_target') and bot.current_attack_target:
            bot.client.debug_text_2d(f"Current Target: {bot.current_attack_target}", 
                                    Point2((0.1, 0.30)), None, 14)
            target_3d = Point3((bot.current_attack_target.x, bot.current_attack_target.y, 
                               bot.get_terrain_z_height(bot.current_attack_target)))
            bot.client.debug_sphere_out(target_3d, 2, Point3((255, 0, 0)))
        
        if main_army:
            army_center = main_army.center
            bot.client.debug_text_2d(f"Army Center: {army_center}", 
                                    Point2((0.1, 0.32)), None, 14)
            army_center_3d = Point3((army_center.x, army_center.y, 
                                   bot.get_terrain_z_height(army_center)))
            bot.client.debug_sphere_out(army_center_3d, 3, Point3((0, 255, 0)))

    # Initialize attack state tracking if not present
    if not hasattr(bot, '_attack_commenced_time'):
        bot._attack_commenced_time = 0.0
    
    # Constants for hysteresis (following example pattern)
    STAY_AGGRESSIVE_FOR = 20.0
    
    # If the army is already attacking, decide whether to continue or retreat
    if bot._commenced_attack:
        # Immediately retreat if we enter an early defensive mode
        if is_early_defensive_mode:
            bot._commenced_attack = False
            nearest_base = bot.townhalls.closest_to(main_army.center)
            if nearest_base:
                control_main_army(bot, main_army, nearest_base.position, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
        # Stay aggressive for minimum duration to prevent oscillation
        elif bot.time < bot._attack_commenced_time + STAY_AGGRESSIVE_FOR:
            control_main_army(bot, main_army, attack_target, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
        # Decide whether to retreat based on army ratio
        else:
            fight_result = bot.mediator.can_win_fight(own_units=bot.own_army, enemy_units=bot.enemy_army, timing_adjust=True, good_positioning=True, workers_do_no_damage=True)
            if fight_result in LOSS_MARGINAL_OR_WORSE:
                # Retreat to the closest base if we're outmatched
                bot._commenced_attack = False
                nearest_base = bot.townhalls.closest_to(main_army.center)
                if nearest_base:
                    control_main_army(bot, main_army, nearest_base.position, 
                                    bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
            # Otherwise, stick with current attack target for stability
            else:
                control_main_army(bot, main_army, attack_target, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
    else:
        # Check our fighting capability first
        fight_result = bot.mediator.can_win_fight(own_units=bot.own_army, enemy_units=bot.enemy_army, timing_adjust=True, good_positioning=True, workers_do_no_damage=True)
        
        
        # Attack if we have a clear advantage (decisive victory or better), regardless of defensive flags
        if fight_result in VICTORY_DECISIVE_OR_BETTER:
            control_main_army(bot, main_army, attack_target, 
                            bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
            bot._commenced_attack = True
            bot._attack_commenced_time = bot.time
        # Only consider attacking if build order is complete and not in early defensive mode
        elif not is_early_defensive_mode:
            if (fight_result in TIE_OR_BETTER and not bot._under_attack):
                control_main_army(bot, main_army, attack_target, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
                bot._commenced_attack = True
                bot._attack_commenced_time = bot.time
            else:
                pass
        # When build order isn't complete or in defensive mode, only redirect for major threats
        elif bot._under_attack and bot.enemy_units:
            # Use enhanced threat assessment to avoid bouncing
            enemy_center, _ = cy_find_units_center_mass(bot.enemy_units, 10.0)
            major_threat_info = assess_threat(bot, bot.enemy_units, main_army, return_details=True)
            # Type assertion since we know return_details=True returns dict
            assert isinstance(major_threat_info, dict), "assess_threat with return_details=True should return dict"
            
            # Only pull main army for significant threats
            if (major_threat_info["threat_level"] >= 7 or 
                major_threat_info["response_type"] == "combat_response"):
                control_main_army(bot, main_army, Point2(enemy_center), 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
            # Otherwise let allocated forces handle it
        else:
            pass


def attack_target(bot, main_army_position: Point2) -> Point2:
    """
    Determines where our main attacking units should move toward.
    Uses example's targeting hierarchy with stable army center mass for proximity checks.
    """
    
    # Get stable army center mass (following example pattern)
    attacking_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    if attacking_units:
        own_center_mass, num_own = cy_find_units_center_mass(attacking_units, 10)
        own_center_mass = Point2(own_center_mass)
    else:
        own_center_mass = main_army_position
    
    # 1. Calculate structure position with enhanced stability
    enemy_structure_pos: Point2 = None
    if bot.enemy_structures:
        valid_structures = bot.enemy_structures.filter(lambda s: s.type_id not in COMMON_UNIT_IGNORE_TYPES)
        if not valid_structures:
            valid_structures = bot.enemy_structures
        if valid_structures:
            # Prefer current target if it's still valid and among the closest structures
            if (bot.current_attack_target and 
                valid_structures.filter(lambda s: cy_distance_to_squared(s.position, bot.current_attack_target) < 25.0)):
                # Current target is still among valid structures, keep it for stability
                enemy_structure_pos = bot.current_attack_target
            else:
                # Use standard closest to enemy natural logic
                enemy_structure_pos = cy_closest_to(bot.mediator.get_enemy_nat, valid_structures).position
    
    # 2. Proximity stickiness - "idea here is if we are near enemy structures/production, don't get distracted"
    if (enemy_structure_pos and 
        cy_distance_to_squared(own_center_mass, enemy_structure_pos) < 450.0):
        bot.current_attack_target = enemy_structure_pos
        return enemy_structure_pos
    
    enemy_army = bot.enemy_units.filter(lambda u: 
        u.type_id not in COMMON_UNIT_IGNORE_TYPES 
        and not u.is_structure 
        and not u.is_burrowed 
        and not u.is_cloaked
    )
    enemy_supply = sum(bot.calculate_supply_cost(u.type_id) for u in enemy_army)
    
    if enemy_supply >= 6 and enemy_army:
        enemy_center, _ = cy_find_units_center_mass(enemy_army, 10.0)
        
        all_close_enemy = bot.mediator.get_units_in_range(
            start_points=[Point2(enemy_center)],
            distances=11.5,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=False,
        )[0]
        clustered_supply = sum(bot.calculate_supply_cost(u.type_id) for u in all_close_enemy)
        
        # Only target army if clustered supply is significant
        if clustered_supply >= 7:
            bot.current_attack_target = Point2(enemy_center)
            return bot.current_attack_target
    
    # 4. Structure fallback
    if enemy_structure_pos:
        bot.current_attack_target = enemy_structure_pos
        return enemy_structure_pos
    
    # 5. Expansion cycling when no targets (following example)
    if bot.is_visible(bot.current_attack_target) if bot.current_attack_target else True:
        fallback = fallback_target(bot)
        bot.current_attack_target = fallback
        return fallback
    
    # 6. Keep current target if area not scouted yet
    if bot.current_attack_target:
        return bot.current_attack_target
    
    # 7. Final fallback
    fallback = bot.enemy_start_locations[0] if bot.time < 240.0 else fallback_target(bot)
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


def manage_defensive_unit_roles(bot) -> None:
    """
    Manages units assigned to BASE_DEFENDER roles.
    Returns them to ATTACKING when threats are cleared (proper ARES way).
    """
    defending_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    
    if not defending_units:
        return
    
    # Check if there are still active threats near bases
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases
    
    active_threats = False
    if ground_near or flying_near:
        # Combine ground + air threats
        all_threats = {}
        for key, value in ground_near.items():
            all_threats[key] = value.copy()
        for key, value in flying_near.items():
            all_threats.setdefault(key, set()).update(value)
        
        # Check if any significant threats remain
        for base_location, enemy_tags in all_threats.items():
            enemy_units = bot.enemy_units.tags_in(enemy_tags)
            if enemy_units:
                threat_position, _ = cy_find_units_center_mass(enemy_units, 10.0)
                threat_info = assess_threat(bot, enemy_units, Units([], bot), return_details=True)
                # Type assertion since we know return_details=True returns dict
                assert isinstance(threat_info, dict), "assess_threat with return_details=True should return dict"
                if threat_info["threat_level"] >= 1:  # Even minor threats keep some defenders
                    active_threats = True
                    break
    
    # If no active threats, return BASE_DEFENDER units to ATTACKING (proper ARES way)
    if not active_threats:
        # Individual role assignments back to ATTACKING (like examples)
        for unit in defending_units:
            bot.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)


