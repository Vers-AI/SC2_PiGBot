import numpy as np

from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.data import Race
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
    render_formation_debug,
)
from bot.utilities.intel import get_enemy_intel_quality

from cython_extensions import (
    cy_pick_enemy_target, cy_closest_to, cy_distance_to, cy_distance_to_squared, cy_in_attack_range, cy_find_units_center_mass,
    cy_adjust_moving_formation
)


# Formation + cohesion constants
FORMATION_COHESION_AHEAD_THRESHOLD = 6.0  # Units this far AHEAD of center slow down
FORMATION_COHESION_SPREAD_THRESHOLD = 8.0  # Max allowed distance from army center
FORMATION_UNIT_MULTIPLIER = 2.0   # Spacing for formation adjustment
FORMATION_RETREAT_ANGLE = 0.3     # Diagonal spread for ranged units


def is_near_choke_or_ramp(bot, army_center: Point2) -> bool:
    """
    O(1) check if army is near a choke/ramp using precomputed grid.
    
    When true, formation logic should be skipped to prevent bottlenecks
    where ranged units block melee units from moving through narrow passages.
    
    Args:
        bot: Bot instance with choke_grid attribute (created in on_start)
        army_center: Center of mass of the squad
        
    Returns:
        True if near choke/ramp (skip formation), False otherwise
    """
    x, y = int(army_center.x), int(army_center.y)
    return bot.choke_grid[x, y] > 1.0


def is_blind_ramp_attack(bot, unit: Unit) -> bool:
    """
    Check if unit is at the bottom of a ramp with enemy ranged units at the
    unseen top - attacking would be a blind disadvantage.
    
    From anglerbot: Used to trigger KeepUnitSafe instead of attacking.
    
    Returns:
        True if unit should NOT attack (retreat instead), False otherwise
    """
    for ramp in bot.game_info.map_ramps:
        # Check if unit is at bottom of this ramp
        if cy_distance_to(unit.position, ramp.bottom_center) > 5.0:
            continue
            
        # Skip if we can see the top - no vision disadvantage
        if bot.is_visible(ramp.top_center):
            continue
            
        # Unit is at bottom, can't see top - check for enemy ranged at top
        # Use all_enemy_units (includes memory) since we can't see them if top is dark
        for enemy in bot.all_enemy_units:
            if enemy.ground_range < 2:  # Skip melee
                continue
            if cy_distance_to(enemy.position, ramp.top_center) < 6.0:
                return True  # Enemy ranged at top - don't attack blind
                
    return False


def is_unit_on_ramp(bot, unit: Unit) -> bool:
    """
    Check if unit is on or near a ramp.
    
    When true and no enemies nearby, unit should move through the ramp
    instead of standing there (even to attack structures).
    
    Returns:
        True if unit is on/near any ramp
    """
    for ramp in bot.game_info.map_ramps:
        # Check if unit is near top or bottom of ramp
        if cy_distance_to(unit.position, ramp.top_center) < 5.0:
            return True
        if cy_distance_to(unit.position, ramp.bottom_center) < 5.0:
            return True
                
    return False


def get_formation_move_target(
    unit: Unit, 
    squad_units: list[Unit], 
    target: Point2,
    bot=None,
) -> Point2:
    """
    Get move target that maintains formation AND cohesion during map crossing.
    
    Formation: Uses cy_adjust_moving_formation to keep melee in front, ranged behind
    Cohesion: Slows down units that get too far ahead OR too spread out
    
    SKIP formation when near chokes/ramps to prevent bottlenecks where ranged
    units block melee from moving through narrow passages.
    
    Args:
        unit: The unit to get move target for
        squad_units: All units in the squad (for center mass calculation)
        target: The ultimate destination
        bot: Bot instance for choke/ramp detection (optional)
        
    Returns:
        Adjusted move target (may be army center if unit is too far ahead/spread)
    """
    if len(squad_units) < 2:
        return target
    
    # Get army center of mass
    army_center, _ = cy_find_units_center_mass(squad_units, 10.0)
    army_center = Point2(army_center)
    
    # Skip formation logic near chokes/ramps to prevent bottlenecks
    if bot is not None and is_near_choke_or_ramp(bot, army_center):
        return target
    
    # Identify melee units (fodder - should be in front)
    fodder_tags = [u.tag for u in squad_units if u.ground_range <= MELEE_RANGE_THRESHOLD]
    
    # Only use cy_adjust_moving_formation if we have melee units (otherwise it does nothing)
    if fodder_tags:
        # Get formation adjustments from cython function
        # Returns dict of {unit_tag: (x, y)} for units that need to move back
        adjusted_positions = cy_adjust_moving_formation(
            our_units=squad_units,
            target=target,
            fodder_tags=fodder_tags,
            unit_multiplier=FORMATION_UNIT_MULTIPLIER,
            retreat_angle=FORMATION_RETREAT_ANGLE
        )
        
        # If this unit needs formation adjustment (ranged unit too far forward), use that
        if unit.tag in adjusted_positions:
            return Point2(adjusted_positions[unit.tag])
    
    # Cohesion check 1: Is this unit too far ahead of army center (toward target)?
    unit_dist_to_target = cy_distance_to(unit.position, target)
    center_dist_to_target = cy_distance_to(army_center, target)
    ahead_distance = center_dist_to_target - unit_dist_to_target
    
    if ahead_distance > FORMATION_COHESION_AHEAD_THRESHOLD:
        # Unit is streaming ahead - move toward army center to wait for others
        return army_center.towards(target, FORMATION_COHESION_AHEAD_THRESHOLD * 0.5)
    
    # Cohesion check 2: Is this unit too far from army center (spread out)?
    dist_from_center = cy_distance_to(unit.position, army_center)
    if dist_from_center > FORMATION_COHESION_SPREAD_THRESHOLD:
        # Unit is too spread out - move toward army center
        return army_center.towards(target, FORMATION_COHESION_SPREAD_THRESHOLD * 0.5)
    
    # Unit is in good position - continue to target
    return target


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
                
                # Ramp safety: don't attack up ramp without vision (blind disadvantage)
                if is_blind_ramp_attack(bot, r_unit):
                    retreat_maneuver = CombatManeuver()
                    retreat_maneuver.add(KeepUnitSafe(unit=r_unit, grid=avoid_grid))
                    bot.register_behavior(retreat_maneuver)
                    continue
                    
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
                    avoid_grid=avoid_grid,
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
                # Skip HTs that should be merging to archons (2+ HTs exist)
                ht_count = sum(1 for c in other_casters if c.type_id == UnitTypeId.HIGHTEMPLAR)
                for caster in other_casters:
                    # Skip HTs when archon merge is possible - let macro.py handle them
                    if caster.type_id == UnitTypeId.HIGHTEMPLAR and ht_count >= 2:
                        continue
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
            
            # Debug: render formation visualization (only during safe map crossing)
            if not close_by and not nearby_structures:
                render_formation_debug(bot, units, move_to)
            
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
                should_wait = False  # Flag to skip fallback movement when unit should wait
                
                # Ramp check - if on ramp with no enemies, push through (don't stand and shoot)
                # This prevents ranged units blocking ramp while attacking structures
                if not close_by and is_unit_on_ramp(bot, unit):
                    no_enemy_maneuver.add(PathUnitToTarget(
                        unit=unit, grid=unit_grid, target=move_to, success_at_distance=3.0
                    ))
                elif not close_by and not nearby_structures:
                    # Safe map crossing - use formation-aware movement to prevent streaming
                    if cy_distance_to_squared(unit.position, move_to) > MAP_CROSSING_DISTANCE_SQ:
                        # Get formation-adjusted target (keeps melee front, ranged back, army together)
                        # Pass bot for choke/ramp detection to prevent bottlenecks
                        formation_target = get_formation_move_target(unit, units, move_to, bot=bot)
                        
                        # If formation_target != move_to, unit is ahead and should WAIT (not move)
                        if formation_target != move_to:
                            # Unit should wait - don't give any move command, let it stop naturally
                            should_wait = True
                            # Optional: give a hold position to make the wait explicit
                            unit.hold_position()
                        else:
                            # Unit is in good position - continue to target
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
                    
                # Fallback for units without orders (but NOT if they should be waiting)
                if not unit.orders and not should_wait:
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
            # Determine fallback direction based on opponent race
            # PvP (main ramp): fall back towards start location
            # PvZ/Random (natural choke): fall back towards natural
            if bot.enemy_race == Race.Protoss:
                fallback_target = bot.start_location
            else:
                fallback_target = bot.natural_expansion
            
            direction = fallback_target - gate_keep_pos
            if direction.length > 0:  # Avoid division by zero
                direction = direction.normalized
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
        
        # === INTEL QUALITY GATE ===
        # Don't trust combat sim if we haven't seen enemy army or info is too stale
        intel = get_enemy_intel_quality(bot)
        
        # Gate 1: Must have seen enemy army at least once
        if not intel["has_intel"]:
            # Combat sim has ZERO enemy data - don't trust it, stay defensive
            return select_defensive_anchor(bot, main_army)
        
        # Gate 2: Adjust required victory threshold based on intel freshness
        # Fresh (>0.7): trust combat sim normally
        # Stale (0.3-0.7): require higher confidence
        # Very stale (<0.3): don't initiate attacks, need fresh intel
        FRESH_INTEL_THRESHOLD = 0.7
        STALE_INTEL_THRESHOLD = 0.3
        
        if intel["freshness"] < STALE_INTEL_THRESHOLD:
            # Intel is very stale - don't initiate attack, stay defensive until we scout
            return select_defensive_anchor(bot, main_army)
        
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
        
        # Adjust required threshold based on intel freshness
        if intel["freshness"] >= FRESH_INTEL_THRESHOLD:
            # Fresh intel - trust combat sim, attack with decisive advantage
            if fight_result in VICTORY_DECISIVE_OR_BETTER:
                bot._commenced_attack = True
                bot._attack_commenced_time = bot.time
                return attack_target
            # With build order complete, attack with marginal advantage
            elif not is_early_defensive_mode:
                if (fight_result in VICTORY_MARGINAL_OR_BETTER and not bot._under_attack):
                    bot._commenced_attack = True
                    bot._attack_commenced_time = bot.time
                    return attack_target
        else:
            # Moderately stale intel (0.3-0.7) - require emphatic victory to initiate
            from ares.consts import VICTORY_EMPHATIC_OR_BETTER
            if fight_result in VICTORY_EMPHATIC_OR_BETTER:
                bot._commenced_attack = True
                bot._attack_commenced_time = bot.time
                return attack_target
        
        # When build order isn't complete or in defensive mode, only redirect for major threats
        if bot._under_attack and bot.enemy_units:
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


def control_defenders(bot) -> None:
    """
    Orchestration function called from bot.py on_step.
    Gets all BASE_DEFENDER units and controls them every frame.
    """
    defenders = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    if not defenders:
        return
    
    # Get threat position set by threat_detection in reactions.py
    threat_position = getattr(bot, '_defender_threat_position', None)
    if threat_position is None:
        # No active threat - move defenders to rally point
        threat_position = bot.main_base_ramp.top_center if bot.main_base_ramp else bot.start_location
    
    control_base_defenders(bot, defenders, threat_position)


def control_base_defenders(bot, defender_units: Units, threat_position: Point2) -> None:
    """
    Controls BASE_DEFENDER units using individual behaviors with squad formation.
    Uses the same micro functions as main army for consistent combat behavior.
    """
    # Wrap in try-except to handle ARES squad manager KeyError when units die mid-frame
    try:
        defensive_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)
    except KeyError:
        # ARES internal tracking got out of sync - unit died but tag wasn't cleaned up
        # Fall back to using defender_units directly without squad grouping
        defensive_squads = None
    
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
        
        if not all_close:
            # No enemies - move towards threat position
            for unit in units:
                move_maneuver = CombatManeuver()
                move_maneuver.add(AMove(unit=unit, target=threat_position))
                bot.register_behavior(move_maneuver)
            continue
        
        # Get priority targets
        priority_targets = get_priority_targets(all_close)
        
        # Separate unit types for appropriate micro
        melee = [u for u in units if u.ground_range <= MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack]
        ranged = [u for u in units if u.ground_range > MELEE_RANGE_THRESHOLD and u.energy == 0 and u.can_attack]
        spellcasters = [u for u in units if u.energy > 0 or u.type_id == UnitTypeId.DISRUPTOR]
        
        # Ranged micro - use same function as main army (aggressive=True for base defense)
        for r_unit in ranged:
            ranged_maneuver = micro_ranged_unit(
                unit=r_unit,
                enemies=all_close,
                priority_targets=priority_targets,
                grid=grid,
                avoid_grid=avoid_grid,
                aggressive=True  # Always aggressive when defending base
            )
            bot.register_behavior(ranged_maneuver)
        
        # Melee micro - use same function as main army
        for m_unit in melee:
            melee_maneuver = micro_melee_unit(
                unit=m_unit,
                enemies=all_close,
                priority_targets=priority_targets,
                avoid_grid=avoid_grid,
                fallback_position=threat_position,
                aggressive=True  # Always aggressive when defending base
            )
            bot.register_behavior(melee_maneuver)
        
        # Spellcasters - stay safe, move with defenders
        for caster in spellcasters:
            caster_maneuver = CombatManeuver()
            caster_maneuver.add(KeepUnitSafe(unit=caster, grid=avoid_grid))
            caster_maneuver.add(AMove(unit=caster, target=threat_position))
            bot.register_behavior(caster_maneuver)


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
            # Position behind gatekeeper (direction depends on opponent race)
            # PvP (main ramp): position towards start location
            # PvZ/Random (natural choke): position towards natural
            if bot.enemy_race == Race.Protoss:
                new_anchor = bot.gatekeeping_pos.towards(bot.start_location, 4)
            else:
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
    Release BASE_DEFENDER units back to ATTACKING when threats clear.
    
    Uses ARES enemy tracking which has hysteresis (enemies must leave beyond
    24 units to clear). This prevents role oscillation from brief visibility gaps.
    """
    defending_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    
    if not defending_units:
        return
    
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases
    
    # Check if ARES is tracking any enemies near bases
    active_threats = any(ground_near.values()) or any(flying_near.values())
    
    if not active_threats:
        for unit in defending_units:
            bot.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)


