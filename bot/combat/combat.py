import numpy as np

from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from ares.consts import LOSS_MARGINAL_OR_WORSE, TIE_OR_BETTER, UnitTreeQueryType, EngagementResult, VICTORY_DECISIVE_OR_BETTER, VICTORY_MARGINAL_OR_BETTER, LOSS_OVERWHELMING_OR_WORSE, LOSS_DECISIVE_OR_WORSE, WORKER_TYPES

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, PathUnitToTarget, StutterUnitBack, AMove, ShootTargetInRange
)
from ares.managers.squad_manager import UnitSquad

from ares.consts import UnitRole

from bot.managers.reactions import assess_threat
from bot.constants import (
    ATTACKING_SQUAD_RADIUS,
    DEFENDER_SQUAD_RADIUS,
    COMMON_UNIT_IGNORE_TYPES,
    DISRUPTOR_IGNORE_TYPES,
    MELEE_RANGE_THRESHOLD,
    STAY_AGGRESSIVE_DURATION,
    UNSAFE_GROUND_CHECK_RADIUS,
    DISRUPTOR_SQUAD_FOLLOW_DISTANCE,
    DISRUPTOR_SQUAD_TARGET_DISTANCE,
    STRUCTURE_ATTACK_RANGE,
    PROXIMITY_STICKY_DISTANCE_SQ,
    MAP_CROSSING_DISTANCE_SQ,
    MAP_CROSSING_SUCCESS_DISTANCE,
    SQUAD_ENEMY_DETECTION_RANGE,
    DEFENDER_ENEMY_DETECTION_RANGE,
    GATEKEEPER_DETECTION_RANGE,
    GATEKEEPER_MOVE_DISTANCE,
    WARP_PRISM_FOLLOW_DISTANCE,
    WARP_PRISM_FOLLOW_OFFSET,
    WARP_PRISM_UNIT_CHECK_RANGE,
    WARP_PRISM_DANGER_DISTANCE,
    EARLY_GAME_TIME_LIMIT,
    EARLY_GAME_SAFE_GROUND_CHECK_BASES,
    SIEGE_TANK_SUPPLY_ADVANTAGE_REQUIRED,
    SQUAD_NEARBY_FRIENDLY_RANGE_SQ,
)
from bot.combat.unit_micro import (
    micro_ranged_unit,
    micro_melee_unit,
    micro_disruptor,
    get_priority_targets,
)
from bot.utilities.debug import (
    render_combat_state_overlay,
    render_disruptor_labels,
    render_nova_labels,
    log_nova_error,
)

from cython_extensions import (
    cy_pick_enemy_target, cy_closest_to, cy_distance_to, cy_distance_to_squared, cy_in_attack_range, cy_find_units_center_mass
)




def control_main_army(bot, main_army: Units, target: Point2, squads: list[UnitSquad]) -> Point2:
    """
    Controls the main army's movement and engagement logic using individual unit behaviors.
    """
    # ARES requires get_squads() to be called before get_position_of_main_squad()
    if not squads:
        squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
    
    pos_of_main_squad: Point2 = bot.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
    grid: np.ndarray = bot.mediator.get_ground_grid
    avoid_grid: np.ndarray = bot.mediator.get_ground_avoidance_grid
    
    if main_army:
        bot.total_health_shield_percentage = (
            sum(unit.shield_health_percentage for unit in main_army) / len(main_army)
        )

    for squad in squads:
        squad_position: Point2 = squad.squad_position
        units: list[Unit] = squad.squad_units
        
        # Skip empty squads
        if not units:
            continue

        # Main squad coordination: main squad goes to target, others converge on main squad
        move_to: Point2 = target if squad.main_squad else pos_of_main_squad

        # Find nearby enemy units
        all_close = bot.mediator.get_units_in_range(
            start_points=[squad_position],
            distances=SQUAD_ENEMY_DETECTION_RANGE,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )[0].filter(lambda u: not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES)
        
        if all_close:
            # Defensive mode override: when not attacking, always engage enemies near our bases
            # This prevents retreating from winnable local fights due to unfavorable global situation
            if not bot._commenced_attack:
                can_engage = True
            else:
                # Attacking mode: use localized combat simulation for tactical decisions
                # Squad-level combat sim evaluates if THIS squad can win against THESE enemies
                own_nearby = main_army.filter(
                    lambda u: cy_distance_to_squared(u.position, squad_position) < SQUAD_NEARBY_FRIENDLY_RANGE_SQ
                )
                
                # Filter enemy units to focus on actual combat units for more conservative simulation
                combat_enemies = all_close.filter(
                    lambda u: u.type_id not in WORKER_TYPES and not u.is_structure
                )
                
                squad_fight_result = bot.mediator.can_win_fight(
                    own_units=own_nearby,
                    enemy_units=combat_enemies,
                    workers_do_no_damage=True,
                )
                
                # Initialize squad tracking if first encounter
                squad_id = squad.squad_id
                if squad_id not in bot._squad_engagement_tracker:
                    bot._squad_engagement_tracker[squad_id] = {"can_engage": True}
                
                # Apply engagement hysteresis: different thresholds for starting vs stopping engagement
                # This prevents squads from flip-flopping between aggressive and defensive behavior
                if bot._squad_engagement_tracker[squad_id]["can_engage"]:
                    # Already engaging: only retreat if situation becomes dire
                    if squad_fight_result in LOSS_OVERWHELMING_OR_WORSE:
                        bot._squad_engagement_tracker[squad_id]["can_engage"] = False
                else:
                    # Not engaging: require advantage before committing to fight
                    if squad_fight_result in VICTORY_MARGINAL_OR_BETTER:
                        bot._squad_engagement_tracker[squad_id]["can_engage"] = True
                
                can_engage = bot._squad_engagement_tracker[squad_id]["can_engage"]
            
            # Separate melee, ranged and spell casters
            # Spellcasters: ground units with energy (HT, Sentries) OR Disruptors (by ID)
            # Excludes: air units with energy (Observers)
            melee = [u for u in units if u.ground_range <= MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack]
            ranged = [u for u in units if u.ground_range > MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack]
            spellcasters = [u for u in units if (u.energy > 0 and not u.is_flying) or u.type_id == UnitTypeId.DISRUPTOR]
            
            # Simple picking logic
            enemy_target = cy_pick_enemy_target(all_close)
            
            # ARES safety awareness - avoid unsafe positions when defending (ranged only)
            ranged_on_unsafe_ground = []
            if not bot._commenced_attack and len(bot.townhalls) <= EARLY_GAME_SAFE_GROUND_CHECK_BASES and bot.time < EARLY_GAME_TIME_LIMIT:
                for r_unit in ranged:
                    if not bot.mediator.is_position_safe(grid=grid, position=r_unit.position):
                        ranged_on_unsafe_ground.append(r_unit)
                        # Find safer position instead of hardcoded start_location
                        safe_spot = bot.mediator.find_closest_safe_spot(
                            from_pos=r_unit.position, 
                            grid=grid,
                            radius=UNSAFE_GROUND_CHECK_RADIUS
                        )
                        r_unit.move(safe_spot)
                        
            # Get priority targets
            priority_targets = get_priority_targets(all_close)
            
            # Ranged micro - superior logic with priority targeting
            for r_unit in ranged:
                if r_unit in ranged_on_unsafe_ground:
                    continue  # Skip units retreating to safer ground
                    
                ranged_maneuver = micro_ranged_unit(
                    unit=r_unit,
                    enemies=all_close,
                    priority_targets=priority_targets,
                    grid=grid,
                    avoid_grid=avoid_grid,
                    aggressive=can_engage
                )
                bot.register_behavior(ranged_maneuver)

            # Melee engages directly - simplified approach
            for m_unit in melee:
                melee_maneuver = micro_melee_unit(
                    unit=m_unit,
                    enemies=all_close,
                    priority_targets=priority_targets,
                    fallback_position=enemy_target.position if enemy_target else move_to,
                    aggressive=can_engage
                )
                bot.register_behavior(melee_maneuver)
                
            # Spellcasters 
            if spellcasters:
                disruptors = [spellcaster for spellcaster in spellcasters if spellcaster.type_id == UnitTypeId.DISRUPTOR]
                other_casters = [spellcaster for spellcaster in spellcasters if spellcaster.type_id != UnitTypeId.DISRUPTOR]
                
                # Handle Disruptors with nova logic
                if disruptors:
                    # Filter enemies for disruptors once: exclude workers and broodlings
                    disruptor_targets = all_close.filter(lambda u: u.type_id not in DISRUPTOR_IGNORE_TYPES)
                    
                    nova_manager = bot.nova_manager if hasattr(bot, 'nova_manager') else None
                    
                    for disruptor in disruptors:
                        micro_disruptor(
                            disruptor=disruptor,
                            enemies=disruptor_targets,
                            friendly_units=units,
                            avoid_grid=avoid_grid,
                            bot=bot,
                            nova_manager=nova_manager,
                            squad_position=squad_position
                        )
                
                # Handle other spellcasters (HTs, Sentries) - stay with army, stay safe
                for caster in other_casters:
                    caster_maneuver = CombatManeuver()
                    caster_maneuver.add(KeepUnitSafe(unit=caster, grid=avoid_grid))
                    caster_maneuver.add(AMove(unit=caster, target=move_to))
                    bot.register_behavior(caster_maneuver)
                        
        else:
            # Check for broader enemy presence (including all unit types)
            close_by = bot.mediator.get_units_in_range(
                start_points=[squad_position],
                distances=SQUAD_ENEMY_DETECTION_RANGE,
                query_tree=UnitTreeQueryType.AllEnemy,
                return_as_dict=False,
            )[0].filter(lambda u: not u.is_memory and not u.is_structure)
            
            # Check for nearby enemy structures
            nearby_structures = bot.enemy_structures.closer_than(STRUCTURE_ATTACK_RANGE, squad_position)
            
            # Movement logic: PathUnitToTarget only when truly safe
            for unit in units:
                unit_grid = bot.mediator.get_air_grid if unit.is_flying else grid
                if unit.type_id == UnitTypeId.COLOSSUS:
                    unit_grid = bot.mediator.get_climber_grid
                    
                no_enemy_maneuver = CombatManeuver()
                
                # Disruptors follow army target when no enemies nearby (like observers)
                if unit.type_id == UnitTypeId.DISRUPTOR:
                    distance_to_target = cy_distance_to(unit.position, move_to)
                    
                    # If far from target, move towards it
                    if distance_to_target > DISRUPTOR_SQUAD_FOLLOW_DISTANCE:
                        no_enemy_maneuver.add(PathUnitToTarget(
                            unit=unit, grid=grid, target=move_to,
                            success_at_distance=DISRUPTOR_SQUAD_TARGET_DISTANCE
                        ))
                    # Otherwise stay put (close enough to target)
                    bot.register_behavior(no_enemy_maneuver)
                    continue  # Skip rest of movement logic for disruptors
                
                # Use PathUnitToTarget only when no enemies or structures nearby (map crossing)
                if not close_by and not nearby_structures:
                    # Safe map crossing - use efficient pathing
                    if cy_distance_to_squared(unit.position, move_to) > MAP_CROSSING_DISTANCE_SQ:
                        no_enemy_maneuver.add(PathUnitToTarget(
                            unit=unit, grid=unit_grid, target=move_to, success_at_distance=MAP_CROSSING_SUCCESS_DISTANCE
                        ))
                elif nearby_structures:
                    # Structures nearby - attack them with AMove for base clearing
                    closest_structure = cy_closest_to(unit.position, nearby_structures)
                    no_enemy_maneuver.add(AMove(unit=unit, target=closest_structure.position))
                else:
                    # Default movement when enemies around but no structures
                    no_enemy_maneuver.add(AMove(unit=unit, target=move_to))
                    
                # Fallback for units without orders
                if not unit.orders:
                    no_enemy_maneuver.add(AMove(unit=unit, target=move_to))
                bot.register_behavior(no_enemy_maneuver)

    return pos_of_main_squad

def gatekeeper_control(bot, gatekeeper: Units) -> None:
    gate_keep_pos = bot.gatekeeping_pos
    any_close = bot.mediator.get_units_in_range(
        start_points=[gate_keep_pos],
        distances=GATEKEEPER_DETECTION_RANGE,
        query_tree=UnitTreeQueryType.EnemyGround,
        return_as_dict=False,
    )[0]
    

    
    if not gate_keep_pos:
        return
    for gate in gatekeeper:
        dist_to_gate: float = cy_distance_to_squared(gate.position, gate_keep_pos)
        if any_close:
            if dist_to_gate != 0:
                gate.move(gate_keep_pos)
                gate(AbilityId.HOLDPOSITION, queue=True)
            else:
                gate(AbilityId.HOLDPOSITION)
        else:
            # Move distance towards the natural expansion
            direction = bot.natural_expansion - gate_keep_pos
            if direction.length > 0:  # Avoid division by zero
                # Normalize to get direction vector with length 1
                direction = direction.normalized
                # Calculate position from gate_keep_pos towards natural
                target_pos = gate_keep_pos + (direction * GATEKEEPER_MOVE_DISTANCE)
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
            if distance_to_center < WARP_PRISM_FOLLOW_DISTANCE:
                prism(AbilityId.MORPH_WARPPRISMPHASINGMODE)
            else:
                # If no warpin in progress, revert to Transport
                # Or simply path near the army
                not_ready_units = [unit for unit in bot.units if not unit.is_ready and unit.distance_to(prism) < WARP_PRISM_UNIT_CHECK_RANGE]
                if prism.type_id == UnitTypeId.WARPPRISMPHASING and not not_ready_units:
                        prism(AbilityId.MORPH_WARPPRISMTRANSPORTMODE)

                elif prism.type_id == UnitTypeId.WARPPRISM:
                    # Keep prism at offset distance behind the army center
                    direction_vector = (prism.position - main_army.center).normalized
                    new_target = main_army.center + direction_vector * WARP_PRISM_FOLLOW_OFFSET
                    maneuver.add(PathUnitToTarget(unit=prism, target=new_target, grid=air_grid, danger_distance=WARP_PRISM_DANGER_DISTANCE))
        else:
            # If no main army, just retreat to natural (or wherever)
            maneuver.add(
                PathUnitToTarget(
                    unit=prism,
                    target=bot.natural_expansion,
                    grid=air_grid,
                    danger_distance=WARP_PRISM_DANGER_DISTANCE
                )
            )

    bot.register_behavior(maneuver)


def handle_attack_toggles(bot, main_army: Units, attack_target: Point2) -> Point2:
    """
    Enhanced attack toggles logic with smart threat response.
    
    Decides when to attack and when to retreat based on relative army strength
    and intelligent threat assessment that prevents bouncing.
    
    Returns the target position where army should move (attack target or retreat position).
    DOES NOT control units - only sets flags and returns target.
    """
    
    # Check if main army is currently handling defense (set by threat_detection)
    if hasattr(bot, '_main_army_defending') and bot._main_army_defending and not bot._under_attack:
        # Main army is handling a major threat, don't interfere - return current attack target
        return attack_target
    
    # Assess the threat level of enemy units for main combat decisions
    enemy_threat_level = assess_threat(bot, bot.enemy_units, main_army)  # Returns simple int
    # Type assertion since return_details=False (default) returns int
    assert isinstance(enemy_threat_level, int), "assess_threat without return_details should return int"

    enemy_army = bot.enemy_army
    # Early game safety - don't attack during cheese reactions
    is_early_defensive_mode = bot._used_cheese_response
    # Only clear early defensive mode when the cheese reaction is completed
    if (is_early_defensive_mode and bot._cheese_reaction_completed) or bot.game_state == 1:  # mid game
        is_early_defensive_mode = False
    
    # Debug visualization (controlled by bot.debug flag)
    render_combat_state_overlay(bot, main_army, enemy_threat_level, is_early_defensive_mode)
    render_disruptor_labels(bot)
    render_nova_labels(bot, getattr(bot, 'nova_manager', None))

    # Siege tank special case: Combat simulator underestimates siege tanks due to splash damage
    # and positional advantage, so require overwhelming force before engaging them
    if bot.enemy_units({UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED}):
        own_supply = sum(bot.calculate_supply_cost(u.type_id) for u in main_army)
        enemy_supply = sum(bot.calculate_supply_cost(u.type_id) for u in bot.enemy_army)
        if own_supply < enemy_supply * SIEGE_TANK_SUPPLY_ADVANTAGE_REQUIRED:
            bot._commenced_attack = False
            if bot.townhalls:
                nearest_base = bot.townhalls.closest_to(main_army.center)
                return nearest_base.position
            else:
                return bot.start_location
    
    # Evaluate current attack state with engagement hysteresis
    if bot._commenced_attack:
        # Maintain commitment for minimum duration to prevent rapid oscillation
        if bot.time < bot._attack_commenced_time + STAY_AGGRESSIVE_DURATION:
            return attack_target
        # After minimum duration, check if we should retreat
        # For early defensive mode (cheese), check for active engagement before retreating
        elif is_early_defensive_mode:
            # Don't immediately retreat if we're actively engaged with enemies nearby
            # This prevents bouncing during cheese reactions
            enemies_near_army = bot.mediator.get_units_in_range(
                start_points=[main_army.center],
                distances=15,
                query_tree=UnitTreeQueryType.AllEnemy,
                return_as_dict=False,
            )[0].filter(lambda u: not u.is_memory and not u.is_structure)
            
            if enemies_near_army:
                # In active combat - maintain current engagement for stability
                return attack_target
            else:
                # No nearby enemies - safe to retreat
                bot._commenced_attack = False
                if bot.townhalls:
                    nearest_base = bot.townhalls.closest_to(main_army.center)
                    return nearest_base.position
                else:
                    return bot.start_location
        # For normal mode, re-evaluate fight result after minimum duration
        else:
            # Filter enemy units to exclude workers and structures for more conservative combat simulation
            combat_enemy_units = [
                u for u in bot.mediator.get_cached_enemy_army
                if u.type_id not in WORKER_TYPES and not u.is_structure
            ]
            fight_result = bot.mediator.can_win_fight(
                own_units=main_army,
                enemy_units=combat_enemy_units,
                workers_do_no_damage=True,
            )
            # Disengage if situation looks bad (decisive loss or worse)
            if fight_result in LOSS_DECISIVE_OR_WORSE:
                bot._commenced_attack = False
                if bot.townhalls:
                    nearest_base = bot.townhalls.closest_to(main_army.center)
                    return nearest_base.position
                else:
                    return bot.start_location
            else:
                return attack_target
    else:
        # Not currently attacking: evaluate if we should initiate attack
        # Filter enemy units to exclude workers and structures for more conservative combat simulation
        combat_enemy_units = [
            u for u in bot.mediator.get_cached_enemy_army
            if u.type_id not in WORKER_TYPES and not u.is_structure
        ]
        fight_result = bot.mediator.can_win_fight(
            own_units=main_army,
            enemy_units=combat_enemy_units,
            workers_do_no_damage=True,
        )
        
        # Initiate attack with decisive advantage regardless of defensive flags
        if fight_result in VICTORY_DECISIVE_OR_BETTER:
            bot._commenced_attack = True
            bot._attack_commenced_time = bot.time
            return attack_target
        # With build order complete, attack with marginal advantage (stricter threshold than maintaining engagement)
        elif not is_early_defensive_mode:
            if (fight_result in VICTORY_MARGINAL_OR_BETTER and not bot._under_attack):
                bot._commenced_attack = True
                bot._attack_commenced_time = bot.time
                return attack_target
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
                return Point2(enemy_center)
        
        # Default: use strategic anchor positioning (smart defensive placement)
        return select_defensive_anchor(bot, main_army)


def attack_target(bot, main_army_position: Point2) -> Point2:
    """
    Determines where our main attacking units should move toward.
    Uses targeting hierarchy with stable army center mass for proximity checks.
    """
    
    # Get stable army center mass
    attacking_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    if attacking_units:
        own_center_mass, num_own = cy_find_units_center_mass(attacking_units, 10)
        own_center_mass = Point2(own_center_mass)
    else:
        own_center_mass = main_army_position
    
    # 1. Calculate structure position with enhanced stability
    enemy_structure_pos: Point2 | None = None
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
        cy_distance_to_squared(own_center_mass, enemy_structure_pos) < PROXIMITY_STICKY_DISTANCE_SQ):
        bot.current_attack_target = enemy_structure_pos
        return enemy_structure_pos
    
    # 3. Army targeting with supply thresholds
    enemy_army = bot.enemy_units.filter(lambda u: 
        u.type_id not in COMMON_UNIT_IGNORE_TYPES 
        and not u.is_structure 
        and not u.is_burrowed 
        and not u.is_cloaked
    )
    enemy_supply = sum(bot.calculate_supply_cost(u.type_id) for u in enemy_army)
    
    # Only prioritize army if substantial
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
    
    # 5. Expansion cycling when no targets
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
    Cycles through expansion locations with mineral field validation.
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


def control_base_defenders(bot, defender_units: Units, threat_position: Point2) -> None:
    """
    Controls BASE_DEFENDER units using individual behaviors with squad formation.
    Uses smaller squad radius for close base defense coordination.
    """
    defensive_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)
    
    if not defensive_squads:
        return
        
    grid: np.ndarray = bot.mediator.get_ground_grid
    avoid_grid: np.ndarray = bot.mediator.get_ground_avoidance_grid
    
    for squad in defensive_squads:
        squad_position: Point2 = squad.squad_position
        units: list[Unit] = squad.squad_units
        
        # Skip empty squads
        if not units:
            continue
        
        # Find nearby enemy units around this defensive squad
        all_close = bot.mediator.get_units_in_range(
            start_points=[squad_position],
            distances=DEFENDER_ENEMY_DETECTION_RANGE,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )[0].filter(lambda u: not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES)
        
        # Get priority targets
        priority_targets = get_priority_targets(all_close)
        
        # Process each defender individually with improved logic
        for unit in units:
            unit_grid = bot.mediator.get_air_grid if unit.is_flying else grid
            
            defending_maneuver = CombatManeuver()
            defending_maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            
            # Handle zealots simply (special case - just AMove)
            if unit.type_id == UnitTypeId.ZEALOT:
                defending_maneuver.add(AMove(unit=unit, target=threat_position))
                bot.register_behavior(defending_maneuver)
                continue
            
            # Separate ranged from melee for better micro
            if unit.ground_range > MELEE_RANGE_THRESHOLD:
                # Ranged defender micro (like main army)
                closest_enemy = cy_closest_to(unit.position, all_close) if all_close else None
                
                if closest_enemy and not unit.weapon_ready:
                    defending_maneuver.add(StutterUnitBack(unit, target=closest_enemy, grid=unit_grid))
                else:
                    # Simplified targeting hierarchy: Priority -> Any -> Move
                    if in_attack_range_priority := cy_in_attack_range(unit, priority_targets):
                        defending_maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_priority))
                    elif in_attack_range_any := cy_in_attack_range(unit, all_close):
                        defending_maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_any))
                    else:
                        # Move towards threats or position
                        if priority_targets:
                            closest_priority = cy_closest_to(unit.position, priority_targets)
                            defending_maneuver.add(AMove(unit=unit, target=closest_priority.position))
                        elif all_close:
                            closest_threat = cy_closest_to(unit.position, all_close)
                            defending_maneuver.add(AMove(unit=unit, target=closest_threat.position))
                        else:
                            defending_maneuver.add(AMove(unit=unit, target=threat_position))
            else:
                # Melee defender micro (like main army)
                if in_attack_range_priority := cy_in_attack_range(unit, priority_targets):
                    defending_maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_priority))
                elif in_attack_range_any := cy_in_attack_range(unit, all_close):
                    defending_maneuver.add(ShootTargetInRange(unit=unit, targets=in_attack_range_any))
                else:
                    # Move towards priority target or threat position
                    if priority_targets:
                        closest_priority = cy_closest_to(unit.position, priority_targets)
                        defending_maneuver.add(AMove(unit=unit, target=closest_priority.position))
                    else:
                        defending_maneuver.add(AMove(unit=unit, target=threat_position))
            
            bot.register_behavior(defending_maneuver)


def select_defensive_anchor(bot, main_army: Units) -> Point2:
    """
    Selects the best strategic anchor position for army placement when not attacking.
    Uses predefined anchors with 15-second cooldown to prevent oscillation.
    
    Priority cascade:
    1. Mid/late game OR 2+ bases: Position at base closest to enemy
    2. Gatekeeper exists: Position behind gatekeeper (avoids bunching)
    3. Structures at natural: Position forward of natural
    4. Fallback: Position back from main ramp
    """
    ANCHOR_CHANGE_COOLDOWN = 15.0  # Seconds between anchor changes
    NATURAL_STRUCTURE_RADIUS = 400.0  # 20 units radius for structure check
    
    # Check if we're still in cooldown (prevent rapid switching)
    if bot._current_defensive_anchor is not None:
        if bot.time < bot._anchor_change_time + ANCHOR_CHANGE_COOLDOWN:
            return bot._current_defensive_anchor
    
    # Determine new anchor position
    new_anchor = None
    
    # PHASE 1: Mid/late game OR successful early expansion (>= 2 ready townhalls)
    if bot.game_state >= 1 or len(bot.townhalls.ready) >= 2:
        # Multi-base coverage mode: position at base closest to enemy
        if bot.townhalls.ready:
            enemy_start = bot.enemy_start_locations[0]
            closest_base = bot.townhalls.ready.closest_to(enemy_start)
            # Offset 3 units towards enemy start for forward positioning
            new_anchor = closest_base.position.towards(enemy_start, 3)
    
    # PHASE 2: Early game single-base positioning
    else:
        # Check for gatekeeper position (best choke control)
        if hasattr(bot, 'gatekeeping_pos') and bot.gatekeeping_pos is not None:
            # Position 4 units behind gatekeeper towards natural (avoid bunching at choke)
            new_anchor = bot.gatekeeping_pos.towards(bot.natural_expansion, 4)
        
        # Check for any structures at natural (shows operational investment)
        elif bot.structures:
            structures_at_natural = bot.structures.filter(lambda s:
                cy_distance_to_squared(s.position, bot.natural_expansion) < NATURAL_STRUCTURE_RADIUS
            )
            if structures_at_natural:
                # Position forward of natural towards enemy
                enemy_start = bot.enemy_start_locations[0]
                new_anchor = bot.natural_expansion.towards(enemy_start, 3)
        
        # Fallback: safe position back from main ramp
        if new_anchor is None:
            # Position back from main towards start location (safe fallback)
            main_ramp = bot.main_base_ramp.top_center
            new_anchor = main_ramp.towards(bot.start_location, 3)
    
    # Ensure we always have a valid anchor (final fallback)
    if new_anchor is None:
        new_anchor = bot.start_location
    
    # Update anchor if it changed
    if new_anchor != bot._current_defensive_anchor:
        bot._current_defensive_anchor = new_anchor
        bot._anchor_change_time = bot.time
        
        # Sync rally_point to anchor (single source of truth)
        bot.rally_point = new_anchor
        
        # Update all gateway rally points to new anchor
        for gateway in bot.structures(UnitTypeId.GATEWAY).ready:
            gateway(AbilityId.RALLY_BUILDING, new_anchor)
    
    return new_anchor


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
        # Individual role assignments back to ATTACKING
        for unit in defending_units:
            bot.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)


