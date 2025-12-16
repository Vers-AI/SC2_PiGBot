"""Combat debug visualization utilities.

Purpose: Centralized debug rendering for combat system, controlled by bot.debug flag
Key Decisions: All debug calls gated by bot.debug, minimal performance impact when disabled
Limitations: Requires bot.debug=True in bot.py to activate
"""

from sc2.position import Point2, Point3
from sc2.units import Units
from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole

from bot.utilities.intel import get_enemy_intel_quality
from cython_extensions import cy_find_units_center_mass, cy_distance_to

# Worker types to filter from combat sim
WORKER_TYPES = {UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE}


def _get_unit_type_summary(units, max_types: int = 3) -> str:
    """Get a compact summary of unit types (e.g., '5Stlk 3Immo 2Colos').
    
    Args:
        units: List or Units collection to summarize
        max_types: Maximum number of unit types to show
        
    Returns:
        Compact string like '5Stlk 3Immo 2Colo'
    """
    from collections import Counter
    
    if not units:
        return "none"
    
    # Count unit types
    type_counts = Counter(u.type_id for u in units)
    
    # Get top N by count
    top_types = type_counts.most_common(max_types)
    
    # Abbreviate unit names (first 4 chars of name, capitalize)
    parts = []
    for unit_type, count in top_types:
        # Get short name from type_id
        name = unit_type.name[:4].title()
        parts.append(f"{count}{name}")
    
    return " ".join(parts)


def render_combat_state_overlay(bot, main_army: Units, enemy_threat_level: int, is_early_defensive_mode: bool) -> None:
    """
    Render on-screen debug text showing combat state, roles, and squad information.
    
    Only renders if bot.debug is True.
    
    Args:
        bot: Bot instance
        main_army: Main army units
        enemy_threat_level: Current enemy threat assessment
        is_early_defensive_mode: Whether in early defensive mode
    """
    if not bot.debug:
        return
    
    # Combat state overlay
    bot.client.debug_text_2d(
        f"Attack: {bot._commenced_attack} Threat: {enemy_threat_level} Under Attack: {bot._under_attack}", 
        Point2((0.1, 0.22)), None, 14
    )
    bot.client.debug_text_2d(
        f"EarlyDefMode: {is_early_defensive_mode} Cheese: {bot._used_cheese_response}", 
        Point2((0.1, 0.24)), None, 14
    )
    
    # Role counts
    attacking_units = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
    defending_units = bot.mediator.get_units_from_role(role=UnitRole.DEFENDING) 
    base_defender_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    
    bot.client.debug_text_2d(
        f"ROLES: ATK:{len(attacking_units)} DEF:{len(defending_units)} BASE:{len(base_defender_units)}", 
        Point2((0.1, 0.26)), None, 14
    )
    
    # Squad counts (use constants for squad radii)
    from bot.constants import ATTACKING_SQUAD_RADIUS, DEFENDER_SQUAD_RADIUS
    attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
    defending_squads = bot.mediator.get_squads(role=UnitRole.DEFENDING, squad_radius=ATTACKING_SQUAD_RADIUS)
    base_defender_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)
    
    bot.client.debug_text_2d(
        f"SQUADS: ATK:{len(attacking_squads)} DEF:{len(defending_squads)} BASE:{len(base_defender_squads)}", 
        Point2((0.1, 0.28)), None, 14
    )
    
    # Army composition info (archon percentage for comp switching)
    if main_army and len(main_army) > 0:
        archon_count = sum(1 for u in main_army if u.type_id == UnitTypeId.ARCHON)
        archon_pct = archon_count / len(main_army)
        bot.client.debug_text_2d(
            f"Archons: {archon_count}/{len(main_army)} ({archon_pct:.0%}) [switch@15%]", 
            Point2((0.1, 0.30)), None, 14
        )
    
    # Combat simulator results
    _render_combat_sim_overlay(bot, main_army)
    
    # Visual markers for targeting
    render_target_markers(bot, main_army)


def _render_combat_sim_overlay(bot, main_army: Units) -> None:
    """Render combat simulator results (global and squad-level)."""
    # Get enemy army (filter workers) - use cached enemy
    cached_enemy = bot.mediator.get_cached_enemy_army or []
    combat_enemies = [u for u in cached_enemy if u.type_id not in WORKER_TYPES]
    
    # Global fight result
    try:
        global_result = bot.mediator.can_win_fight(
            own_units=main_army,
            enemy_units=combat_enemies,
            workers_do_no_damage=True,
        )
        bot.client.debug_text_2d(
            f"Global Fight: {global_result.name}", 
            Point2((0.1, 0.32)), None, 14
        )
    except Exception:
        bot.client.debug_text_2d(
            "Global Fight: N/A", 
            Point2((0.1, 0.32)), None, 14
        )
    
    # Intel quality display
    intel = get_enemy_intel_quality(bot)
    freshness_bar = "█" * int(intel["freshness"] * 10) + "░" * (10 - int(intel["freshness"] * 10))
    intel_status = "FRESH" if intel["freshness"] >= 0.7 else ("STALE" if intel["freshness"] >= 0.3 else "BLIND")
    
    # Intel urgency indicator (shows response thresholds)
    urgency = getattr(bot, '_intel_urgency', 0.0)
    urgency_bar = "█" * int(urgency * 10) + "░" * (10 - int(urgency * 10))
    urgency_status = ""
    if urgency >= 0.7:
        urgency_status = " →UPGRADE"
    elif urgency >= 0.5:
        urgency_status = " →+OBS"
    elif urgency >= 0.3:
        urgency_status = " →PRIORITY"
    
    bot.client.debug_text_2d(
        f"Intel: [{freshness_bar}] {intel_status} ({intel['visible_count']}vis/{intel['memory_count']}mem)", 
        Point2((0.1, 0.30)), None, 12
    )
    bot.client.debug_text_2d(
        f"Urgency: [{urgency_bar}]{urgency_status}", 
        Point2((0.1, 0.28)), None, 12
    )
    
    # Army compositions
    own_types = _get_unit_type_summary(main_army or [])
    enemy_types = _get_unit_type_summary(combat_enemies)
    
    bot.client.debug_text_2d(
        f"Own: {len(main_army) if main_army else 0}u [{own_types}]", 
        Point2((0.1, 0.34)), None, 12
    )
    bot.client.debug_text_2d(
        f"Enemy: {len(combat_enemies)}u [{enemy_types}]", 
        Point2((0.1, 0.36)), None, 12
    )
    
    # Squad engagement tracker summary
    tracker = getattr(bot, '_squad_engagement_tracker', {})
    if tracker:
        engaged = sum(1 for v in tracker.values() if v.get("can_engage", False))
        total = len(tracker)
        bot.client.debug_text_2d(
            f"Squad Engage: {engaged}/{total} squads", 
            Point2((0.1, 0.38)), None, 14
        )
    
    # 3D squad labels at squad positions
    _render_squad_labels(bot, tracker)


def _render_squad_labels(bot, tracker: dict) -> None:
    """Render 3D labels at each attacking squad's position showing engagement status."""
    from bot.constants import ATTACKING_SQUAD_RADIUS
    
    attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
    
    for squad in attacking_squads:
        squad_id = squad.squad_id
        can_engage = tracker.get(squad_id, {}).get("can_engage", True)
        
        # Color: green if engaging, red if retreating
        color = (0, 255, 0) if can_engage else (255, 0, 0)
        status = "ATK" if can_engage else "RTR"
        label = f"S{squad_id[-4:]}:{status}"  # Short squad ID + status
        
        # Draw at squad center
        pos = squad.squad_position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 1.5)),
            color=color,
            size=12,
        )


def render_target_markers(bot, main_army: Units) -> None:
    """
    Render 3D debug spheres showing current attack target and army center.
    
    Only renders if bot.debug is True.
    
    Args:
        bot: Bot instance
        main_army: Main army units
    """
    if not bot.debug:
        return
    
    # Current attack target marker (red sphere)
    if hasattr(bot, 'current_attack_target') and bot.current_attack_target:
        bot.client.debug_text_2d(
            f"Current Target: {bot.current_attack_target}", 
            Point2((0.1, 0.40)), None, 14
        )
        target_3d = Point3((
            bot.current_attack_target.x, 
            bot.current_attack_target.y, 
            bot.get_terrain_z_height(bot.current_attack_target)
        ))
        bot.client.debug_sphere_out(target_3d, 2, Point3((255, 0, 0)))
    
    # Army center marker (green sphere)
    if main_army:
        army_center = main_army.center
        bot.client.debug_text_2d(
            f"Army Center: {army_center}", 
            Point2((0.1, 0.42)), None, 14
        )
        army_center_3d = Point3((
            army_center.x, 
            army_center.y, 
            bot.get_terrain_z_height(army_center)
        ))
        bot.client.debug_sphere_out(army_center_3d, 3, Point3((0, 255, 0)))


def render_disruptor_labels(bot) -> None:
    """Render 3D labels on Disruptors showing ability status and target info."""
    if not bot.debug:
        return
    
    from sc2.ids.ability_id import AbilityId
    
    disruptors = bot.units(UnitTypeId.DISRUPTOR)
    
    for disruptor in disruptors:
        # Check if Nova ability is ready
        can_nova = AbilityId.EFFECT_PURIFICATIONNOVA in disruptor.abilities
        
        # Build label
        if can_nova:
            label = "NOVA READY"
            color = (0, 255, 0)  # Green
        else:
            label = "CD"
            color = (128, 128, 128)  # Gray
        
        # Draw at unit position
        pos = disruptor.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 2.0)),
            color=color,
            size=12,
        )


def render_nova_labels(bot, nova_manager) -> None:
    """Render 3D labels on active Novas showing frames left and target."""
    if not bot.debug or nova_manager is None:
        return
    
    for nova in nova_manager.active_novas:
        if nova.unit is None:
            continue
        
        # Find the nova unit in current game state
        nova_unit = None
        for unit in bot.units:
            if unit.tag == nova.unit.tag:
                nova_unit = unit
                break
        
        if nova_unit is None:
            continue
        
        # Build label with frames left and target
        frames = nova.frames_left
        target = nova.best_target_pos
        
        if target:
            dist = nova_unit.position.distance_to(target)
            label = f"F:{frames} D:{dist:.1f}"
        else:
            label = f"F:{frames}"
        
        # Color based on frames left (green->yellow->red)
        if frames > 30:
            color = (0, 255, 0)
        elif frames > 15:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)
        
        pos = nova_unit.position
        z = bot.get_terrain_z_height(pos)
        bot.client.debug_text_world(
            label,
            Point3((pos.x, pos.y, z + 1.0)),
            color=color,
            size=10,
        )
        
        # Draw line to target
        if target:
            target_z = bot.get_terrain_z_height(target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z)),
                Point3((target.x, target.y, target_z)),
                color=Point3((255, 165, 0)),  # Orange line to target
            )


def render_formation_debug(bot, squad_units: list, target) -> None:
    """
    Render formation/cohesion debug visualization.
    
    Shows:
    - Army center of mass (blue sphere)
    - Cohesion radius (blue circle)
    - Units that are ahead of formation (yellow labels + lines)
    - Units with formation adjustments (cyan labels + lines to adjusted pos)
    """
    if not bot.debug or not squad_units or len(squad_units) < 2:
        return
    
    from bot.combat.combat import (
        get_formation_move_target, 
        FORMATION_COHESION_AHEAD_THRESHOLD,
        FORMATION_COHESION_SPREAD_THRESHOLD,
        MELEE_RANGE_THRESHOLD
    )
    from sc2.position import Point2
    
    # Get army center of mass
    army_center, _ = cy_find_units_center_mass(squad_units, 10.0)
    army_center = Point2(army_center)
    center_z = bot.get_terrain_z_height(army_center)
    
    # Draw army center (blue sphere)
    bot.client.debug_sphere_out(
        Point3((army_center.x, army_center.y, center_z + 0.5)),
        1.5,
        Point3((0, 100, 255))  # Blue
    )
    
    # Draw cohesion spread radius circle (outer - orange)
    import math
    num_points = 24
    for i in range(num_points):
        angle1 = 2 * math.pi * i / num_points
        angle2 = 2 * math.pi * (i + 1) / num_points
        p1 = Point2((army_center.x + FORMATION_COHESION_SPREAD_THRESHOLD * math.cos(angle1),
                     army_center.y + FORMATION_COHESION_SPREAD_THRESHOLD * math.sin(angle1)))
        p2 = Point2((army_center.x + FORMATION_COHESION_SPREAD_THRESHOLD * math.cos(angle2),
                     army_center.y + FORMATION_COHESION_SPREAD_THRESHOLD * math.sin(angle2)))
        z1 = bot.get_terrain_z_height(p1)
        z2 = bot.get_terrain_z_height(p2)
        bot.client.debug_line_out(
            Point3((p1.x, p1.y, z1 + 0.2)),
            Point3((p2.x, p2.y, z2 + 0.2)),
            color=Point3((255, 165, 0))  # Orange circle for spread threshold
        )
    
    # Calculate distances
    center_dist_to_target = cy_distance_to(army_center, target)
    
    # Check each unit's formation status
    ahead_count = 0
    spread_count = 0
    adjusted_count = 0
    
    for unit in squad_units:
        unit_dist_to_target = cy_distance_to(unit.position, target)
        ahead_distance = center_dist_to_target - unit_dist_to_target
        dist_from_center = cy_distance_to(unit.position, army_center)
        
        pos = unit.position
        z = bot.get_terrain_z_height(pos)
        
        # Get this unit's formation target
        formation_target = get_formation_move_target(unit, squad_units, target)
        
        is_melee = unit.ground_range <= MELEE_RANGE_THRESHOLD
        
        # Unit is streaming ahead (cohesion triggered) - will HOLD POSITION
        if ahead_distance > FORMATION_COHESION_AHEAD_THRESHOLD:
            ahead_count += 1
            # Yellow label for units holding/waiting
            label = f"HOLD +{ahead_distance:.1f}"
            bot.client.debug_text_world(
                label,
                Point3((pos.x, pos.y, z + 1.0)),
                color=(255, 255, 0),  # Yellow
                size=10,
            )
            # Draw line to where they're going (army center area)
            ft_z = bot.get_terrain_z_height(formation_target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z + 0.3)),
                Point3((formation_target.x, formation_target.y, ft_z + 0.3)),
                color=Point3((255, 255, 0))  # Yellow
            )
        # Unit is too spread out from center
        elif dist_from_center > FORMATION_COHESION_SPREAD_THRESHOLD:
            spread_count += 1
            # Orange label for spread units
            label = f"SPREAD {dist_from_center:.1f}"
            bot.client.debug_text_world(
                label,
                Point3((pos.x, pos.y, z + 1.0)),
                color=(255, 165, 0),  # Orange
                size=10,
            )
            # Draw line to formation target
            ft_z = bot.get_terrain_z_height(formation_target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z + 0.3)),
                Point3((formation_target.x, formation_target.y, ft_z + 0.3)),
                color=Point3((255, 165, 0))  # Orange
            )
        # Unit got formation adjustment (ranged behind melee)
        elif formation_target != target:
            adjusted_count += 1
            # Cyan label for formation-adjusted units
            label = "FORM" if not is_melee else "FRONT"
            bot.client.debug_text_world(
                label,
                Point3((pos.x, pos.y, z + 1.0)),
                color=(0, 255, 255),  # Cyan
                size=10,
            )
            # Draw line to adjusted position
            ft_z = bot.get_terrain_z_height(formation_target)
            bot.client.debug_line_out(
                Point3((pos.x, pos.y, z + 0.3)),
                Point3((formation_target.x, formation_target.y, ft_z + 0.3)),
                color=Point3((0, 255, 255))  # Cyan
            )
    
    # Summary text
    bot.client.debug_text_2d(
        f"Formation: {ahead_count} ahead, {spread_count} spread, {adjusted_count} form",
        Point2((0.1, 0.44)), None, 14
    )


def render_base_defender_debug(bot) -> None:
    """
    Render debug visualization for BASE_DEFENDER units.
    
    Shows:
    - Per-unit detection range circles (yellow)
    - Detected enemies count
    - Threat position marker (red sphere)
    """
    if not bot.debug:
        return
    
    from bot.constants import UNIT_ENEMY_DETECTION_RANGE, COMMON_UNIT_IGNORE_TYPES
    from ares.consts import UnitTreeQueryType
    import math
    
    # Get BASE_DEFENDER units
    defender_units = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
    
    # Draw threat position (red sphere)
    threat_pos = getattr(bot, '_defender_threat_position', None)
    if threat_pos:
        z = bot.get_terrain_z_height(threat_pos)
        bot.client.debug_sphere_out(
            Point3((threat_pos.x, threat_pos.y, z + 0.5)),
            1.5,
            Point3((255, 0, 0))  # Red
        )
        bot.client.debug_text_world(
            "THREAT",
            Point3((threat_pos.x, threat_pos.y, z + 2.0)),
            color=(255, 0, 0),
            size=12,
        )
    
    if not defender_units:
        return
    
    # Per-unit detection (matches control_base_defenders logic)
    unit_positions = [u.position for u in defender_units]
    all_close_results = bot.mediator.get_units_in_range(
        start_points=unit_positions,
        distances=UNIT_ENEMY_DETECTION_RANGE,
        query_tree=UnitTreeQueryType.AllEnemy,
        return_as_dict=False,
    )
    
    # Combine and deduplicate enemies
    seen_tags = set()
    all_close_list = []
    for result in all_close_results:
        for u in result:
            if u.tag not in seen_tags and not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES:
                seen_tags.add(u.tag)
                all_close_list.append(u)
    
    # Draw detection circle around each defender unit
    num_points = 16
    for unit in defender_units:
        unit_pos = unit.position
        z = bot.get_terrain_z_height(unit_pos)
        
        # Detection range circle (yellow)
        for i in range(num_points):
            angle1 = 2 * math.pi * i / num_points
            angle2 = 2 * math.pi * (i + 1) / num_points
            p1 = Point2((
                unit_pos.x + UNIT_ENEMY_DETECTION_RANGE * math.cos(angle1),
                unit_pos.y + UNIT_ENEMY_DETECTION_RANGE * math.sin(angle1)
            ))
            p2 = Point2((
                unit_pos.x + UNIT_ENEMY_DETECTION_RANGE * math.cos(angle2),
                unit_pos.y + UNIT_ENEMY_DETECTION_RANGE * math.sin(angle2)
            ))
            z1 = bot.get_terrain_z_height(p1)
            z2 = bot.get_terrain_z_height(p2)
            bot.client.debug_line_out(
                Point3((p1.x, p1.y, z1 + 0.2)),
                Point3((p2.x, p2.y, z2 + 0.2)),
                color=Point3((255, 255, 0))  # Yellow
            )
    
    # Summary label at center of defenders
    center = defender_units.center
    z = bot.get_terrain_z_height(center)
    enemy_count = len(all_close_list)
    color = (0, 255, 0) if enemy_count > 0 else (255, 255, 0)
    label = f"DEF:{len(defender_units)}u E:{enemy_count}"
    bot.client.debug_text_world(
        label,
        Point3((center.x, center.y, z + 2.0)),
        color=color,
        size=12,
    )
    
    # Draw lines from defender center to detected enemies (green)
    for enemy in all_close_list:
        enemy_z = bot.get_terrain_z_height(enemy.position)
        bot.client.debug_line_out(
            Point3((center.x, center.y, z + 0.5)),
            Point3((enemy.position.x, enemy.position.y, enemy_z + 0.5)),
            color=Point3((0, 255, 0))  # Green
        )


def log_nova_error(error: Exception) -> None:
    """
    Log NovaManager errors (always shown, not gated by debug flag).
    
    Args:
        error: Exception that occurred
    """
    print(f"DEBUG ERROR updating NovaManager: {error}")
