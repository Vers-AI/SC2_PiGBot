import numpy as np

from sc2.position import Point2, Point3
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
}

# Note: ATTACK_TARGET_IGNORE was redundant - all items already in COMMON_UNIT_IGNORE_TYPES


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
                control_main_army(bot, main_army, threat_position, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
               


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
        
        # Visual debug markers for targeting
        if hasattr(bot, 'current_attack_target') and bot.current_attack_target:
            bot.client.debug_text_2d(f"Current Target: {bot.current_attack_target}", 
                                    Point2((0.1, 0.26)), None, 14)
            target_3d = Point3((bot.current_attack_target.x, bot.current_attack_target.y, 
                               bot.get_terrain_z_height(bot.current_attack_target)))
            bot.client.debug_sphere_out(target_3d, 2, Point3((255, 0, 0)))
        
        if main_army:
            army_center = main_army.center
            bot.client.debug_text_2d(f"Army Center: {army_center}", 
                                    Point2((0.1, 0.28)), None, 14)
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
        if bot.debug:
            print(f"DEBUG ATTACK_TOGGLES: Currently attacking, evaluating continuation...")
        # Immediately retreat if we enter an early defensive mode
        if is_early_defensive_mode:
            if bot.debug:
                print(f"DEBUG ATTACK_TOGGLES: Early defensive mode - RETREATING")
            bot._commenced_attack = False
            nearest_base = bot.townhalls.closest_to(main_army.center)
            if nearest_base:
                control_main_army(bot, main_army, nearest_base.position, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
        # Stay aggressive for minimum duration to prevent oscillation
        elif bot.time < bot._attack_commenced_time + STAY_AGGRESSIVE_FOR:
            if bot.debug:
                print(f"DEBUG ATTACK_TOGGLES: In aggressive timeout period - continuing attack")
            control_main_army(bot, main_army, attack_target, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
        # Decide whether to retreat based on army ratio
        else:
            fight_result = bot.mediator.can_win_fight(own_units=bot.own_army, enemy_units=bot.enemy_army, timing_adjust=True, good_positioning=True, workers_do_no_damage=True)
            if bot.debug:
                print(f"DEBUG ATTACK_TOGGLES: Fight result: {fight_result}, Threat level: {enemy_threat_level}")
            if fight_result in LOSS_MARGINAL_OR_WORSE:
                # Retreat to the closest base if we're outmatched
                if bot.debug:
                    print(f"DEBUG ATTACK_TOGGLES: Fight result poor - RETREATING")
                bot._commenced_attack = False
                nearest_base = bot.townhalls.closest_to(main_army.center)
                if nearest_base:
                    control_main_army(bot, main_army, nearest_base.position, 
                                    bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
            # Only override target if threat is very high AND there are enemy units to fight
            elif enemy_threat_level >= 5 and bot.enemy_army:
                if bot.debug:
                    print(f"DEBUG ATTACK_TOGGLES: High threat detected with enemy army - REDIRECTING to enemy center")
                enemy_center, _ = cy_find_units_center_mass(bot.enemy_army, 10.0)
                control_main_army(bot, main_army, Point2(enemy_center), 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
            # Otherwise, stick with current attack target for stability
            else:
                if bot.debug:
                    print(f"DEBUG ATTACK_TOGGLES: Continuing attack to target: {attack_target}")
                control_main_army(bot, main_army, attack_target, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
    else:
        # Check our fighting capability first
        fight_result = bot.mediator.can_win_fight(own_units=bot.own_army, enemy_units=bot.enemy_army, timing_adjust=True, good_positioning=True, workers_do_no_damage=True)
        
        if bot.debug:
            print(f"DEBUG ATTACK_TOGGLES: Not attacking yet, evaluating start conditions...")
            print(f"DEBUG ATTACK_TOGGLES: Fight result: {fight_result}, Early def mode: {is_early_defensive_mode}, Under attack: {bot._under_attack}")
        
        # Attack if we have a clear advantage (decisive victory or better), regardless of defensive flags
        if fight_result in VICTORY_DECISIVE_OR_BETTER:
            if bot.debug:
                print(f"DEBUG ATTACK_TOGGLES: Decisive victory - STARTING ATTACK to {attack_target}")
            control_main_army(bot, main_army, attack_target, 
                            bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
            bot._commenced_attack = True
            bot._attack_commenced_time = bot.time
        # Only consider attacking if build order is complete and not in early defensive mode
        elif not is_early_defensive_mode:
            if (fight_result in TIE_OR_BETTER and not bot._under_attack):
                if bot.debug:
                    print(f"DEBUG ATTACK_TOGGLES: Good fight odds & not defending - STARTING ATTACK to {attack_target}")
                control_main_army(bot, main_army, attack_target, 
                                bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
                bot._commenced_attack = True
                bot._attack_commenced_time = bot.time
            else:
                if bot.debug:
                    print(f"DEBUG ATTACK_TOGGLES: Not attacking - poor fight odds or under attack")
        # When build order isn't complete or in defensive mode, focus on defending if needed
        elif bot._under_attack and bot.enemy_units:
            if bot.debug:
                print(f"DEBUG ATTACK_TOGGLES: Defensive mode but under attack - DEFENDING at enemy center")
            enemy_center, _ = cy_find_units_center_mass(bot.enemy_units, 10.0)
            control_main_army(bot, main_army, Point2(enemy_center), 
                            bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
        else:
            if bot.debug:
                print(f"DEBUG ATTACK_TOGGLES: Staying passive - early defensive mode or no threats")


def attack_target(bot, main_army_position: Point2) -> Point2:
    """
    Determines where our main attacking units should move toward.
    Uses example's targeting hierarchy with proximity stickiness to prevent bouncing.
    """
    
    if bot.debug:
        print(f"DEBUG TARGETING: Army position: {main_army_position}, Current target: {bot.current_attack_target}")
    
    # 1. Calculate structure position once (following example pattern)
    enemy_structure_pos: Point2 = None
    if bot.enemy_structures:
        valid_structures = bot.enemy_structures.filter(lambda s: s.type_id not in COMMON_UNIT_IGNORE_TYPES)
        if not valid_structures:
            valid_structures = bot.enemy_structures
        if valid_structures:
            enemy_structure_pos = cy_closest_to(bot.mediator.get_enemy_nat, valid_structures).position
    
    # 2. Proximity stickiness - key anti-bouncing mechanism from example
    if (enemy_structure_pos and 
        cy_distance_to_squared(main_army_position, enemy_structure_pos) < 450.0):
        if bot.debug:
            print(f"DEBUG TARGETING: Near structures - sticking to {enemy_structure_pos}")
        bot.current_attack_target = enemy_structure_pos
        return enemy_structure_pos
    
    # 3. Army targeting with supply thresholds (following example)
    enemy_army = bot.enemy_units.filter(lambda u: 
        u.type_id not in COMMON_UNIT_IGNORE_TYPES 
        and not u.is_structure 
        and not u.is_burrowed 
        and not u.is_cloaked
    )
    enemy_supply = sum(bot.calculate_supply_cost(u.type_id) for u in enemy_army)
    
    if bot.debug:
        print(f"DEBUG TARGETING: Enemy army supply: {enemy_supply}")
    
    # Only prioritize army if substantial (following example thresholds)
    if enemy_supply >= 6 and enemy_army:
        enemy_center, _ = cy_find_units_center_mass(enemy_army, 10.0)
        
        # Check clustered supply around army center
        all_close_enemy = bot.mediator.get_units_in_range(
            start_points=[Point2(enemy_center)],
            distances=11.5,
            query_tree=UnitTreeQueryType.EnemyGround,
            return_as_dict=False,
        )[0]
        clustered_supply = sum(bot.calculate_supply_cost(u.type_id) for u in all_close_enemy)
        
        # Only target army if clustered supply is significant
        if clustered_supply >= 7:
            if bot.debug:
                print(f"DEBUG TARGETING: Prioritizing enemy army cluster (supply: {clustered_supply}) at {Point2(enemy_center)}")
            bot.current_attack_target = Point2(enemy_center)
            return bot.current_attack_target
    
    # 4. Structure fallback
    if enemy_structure_pos:
        if bot.debug:
            print(f"DEBUG TARGETING: Targeting structures at {enemy_structure_pos}")
        bot.current_attack_target = enemy_structure_pos
        return enemy_structure_pos
    
    # 5. Expansion cycling when no targets (following example)
    if bot.is_visible(bot.current_attack_target) if bot.current_attack_target else True:
        fallback = fallback_target(bot)
        if bot.debug:
            print(f"DEBUG TARGETING: No targets - cycling to expansion at {fallback}")
        bot.current_attack_target = fallback
        return fallback
    
    # 6. Keep current target if area not scouted yet
    if bot.current_attack_target:
        if bot.debug:
            print(f"DEBUG TARGETING: Keeping current target - area not scouted yet")
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
        control_main_army(bot, main_army, bot.natural_expansion.towards(bot.game_info.map_center, 2), bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=9.0))
        print("Regrouping army")


