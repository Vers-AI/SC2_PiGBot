"""Combat debug visualization utilities.

Purpose: Centralized debug rendering for combat system, controlled by bot.debug flag
Key Decisions: All debug calls gated by bot.debug, minimal performance impact when disabled
Limitations: Requires bot.debug=True in bot.py to activate
"""

from sc2.position import Point2, Point3
from sc2.units import Units
from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole

# Worker types to filter from combat sim
WORKER_TYPES = {UnitTypeId.SCV, UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.MULE}


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
    # Global fight result (army vs army, excluding workers/structures)
    try:
        combat_enemies = [
            u for u in bot.mediator.get_cached_enemy_army
            if u.type_id not in WORKER_TYPES and not u.is_structure
        ]
        global_result = bot.mediator.can_win_fight(
            own_units=bot.own_army,
            enemy_units=combat_enemies,
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
    
    # Squad engagement tracker summary
    tracker = getattr(bot, '_squad_engagement_tracker', {})
    if tracker:
        engaged = sum(1 for v in tracker.values() if v.get("can_engage", False))
        total = len(tracker)
        bot.client.debug_text_2d(
            f"Squad Engage: {engaged}/{total} squads", 
            Point2((0.1, 0.34)), None, 14
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
            Point2((0.1, 0.36)), None, 14
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
            Point2((0.1, 0.38)), None, 14
        )
        army_center_3d = Point3((
            army_center.x, 
            army_center.y, 
            bot.get_terrain_z_height(army_center)
        ))
        bot.client.debug_sphere_out(army_center_3d, 3, Point3((0, 255, 0)))


def log_nova_error(error: Exception) -> None:
    """
    Log NovaManager errors (always shown, not gated by debug flag).
    
    Args:
        error: Exception that occurred
    """
    print(f"DEBUG ERROR updating NovaManager: {error}")
