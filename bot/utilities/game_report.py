"""
Game Report Utility
Handles presentation and formatting of reports:
- Startup report (called once in on_start)
- Periodic intel reports (every 30 game seconds)
- Replay tagging (for filtering replays later)
- End-game performance report (uses PerformanceMonitor)
"""

from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole, WORKER_TYPES
from sc2.data import Race
from bot.managers.macro import get_economy_state


def print_startup_report(bot) -> None:
    """Print one-time startup report with static game information.
    
    Called once in on_start() to log initial game state.
    Always prints (not gated by bot.debug).
    
    Args:
        bot: Bot instance
    """
    print("\n" + "="*60)
    print("  GAME STARTUP REPORT")
    print("="*60)
    print(f"  Map: {bot.game_info.map_name}")
    print(f"  Enemy Race: {bot.enemy_race.name}")
    print(f"  Build Chosen: {bot.build_order_runner.chosen_opening}")
    print(f"  Natural Expansion: {bot.natural_expansion}")
    print(f"  Enemy Natural: {bot.mediator.get_enemy_nat}")
    print(f"  Rush Distance Tier: {bot.rush_distance_tier}")
    rush_time_str = f"{bot._rush_time_seconds:.1f}s" if bot._rush_time_seconds > 0 else "unknown"
    print(f"  Rush Time: {rush_time_str}")
    print("="*60 + "\n")


def print_periodic_intel_report(bot, iteration: int) -> None:
    """Print periodic intelligence report every 30 game seconds.
    
    Shows bot decision-making state for field analysis.
    Always prints (not gated by bot.debug).
    
    Args:
        bot: Bot instance
        iteration: Current game iteration
    """
    # Report every 30 game seconds, starting at 30 seconds
    # Use range-based check to handle step timing variations (robust to frame spikes)
    if bot.time < 30:
        return
    
    # Initialize tracker if needed
    if not hasattr(bot, '_last_intel_report_time'):
        bot._last_intel_report_time = 0.0
    
    # Check if we've crossed a 30-second boundary since last report
    next_report_time = bot._last_intel_report_time + 30.0
    if bot.time < next_report_time:
        return
    
    # Update tracker to the boundary we're reporting for (snap to 30s grid)
    bot._last_intel_report_time = (int(bot.time) // 30) * 30
    
    # Display rounded to nearest 30s boundary
    report_time = bot._last_intel_report_time
    game_time_minutes = int(report_time // 60)
    game_time_seconds = int(report_time % 60)
    
    print("\n" + "="*60)
    print(f"  INTEL REPORT {game_time_minutes}:{game_time_seconds:02d}")
    print("="*60)
    
    # === COMBAT STATUS ===
    print("\n  COMBAT STATUS:")
    print(f"    Attack Commenced: {bot._commenced_attack}")
    print(f"    Under Attack: {bot._under_attack}")
    print(f"    Cheese Response: {bot._used_cheese_response}")
    print(f"    Game State: {bot.game_state} ({'Early' if bot.game_state == 0 else 'Mid' if bot.game_state == 1 else 'Late'})")
    
    # Fight simulation (filter workers for accurate combat assessment)
    try:
        own_combat = [u for u in bot.own_army if u.type_id not in WORKER_TYPES]
        enemy_combat = [u for u in bot.enemy_army if u.type_id not in WORKER_TYPES]
        fight_result = bot.mediator.can_win_fight(
            own_units=own_combat,
            enemy_units=enemy_combat,
            timing_adjust=True,
            good_positioning=True,
            workers_do_no_damage=True
        )
        print(f"    Can Win Fight: {fight_result}")
    except Exception as e:
        print(f"    Can Win Fight: Error ({e})")
    
    # === ARMY COMPOSITION ===
    print("\n  ARMY COMPOSITION:")
    _print_army_composition(bot)
    
    # === UNIT ROLES ===
    print("\n  UNIT ROLES:")
    _print_unit_roles(bot)
    
    # === ENEMY INTELLIGENCE ===
    print("\n  ENEMY INTELLIGENCE:")
    _print_enemy_intel(bot)
    
    # === ECONOMY ===
    print("\n  ECONOMY:")
    _print_economy_state(bot)
    
    # === TARGET & POSITIONING ===
    if hasattr(bot, 'current_attack_target') and bot.current_attack_target:
        print("\n  TARGETING:")
        print(f"    Current Target: {bot.current_attack_target}")
    
    # === RUSH DETECTION (Zerg only) ===
    if bot.enemy_race in {Race.Zerg, Race.Random}:
        _print_rush_detection_status(bot)
    
    print("="*60 + "\n")


def get_replay_tags_to_send(bot) -> list[str]:
    """
    Collect replay tags that should be sent this iteration.
    Returns list of tags to send via chat_send().
    
    Tags are only sent once per game (tracked in bot._replay_tags_sent).
    
    Args:
        bot: Bot instance
        
    Returns:
        List of tag strings to send
    """
    # Initialize tags set if needed
    if not hasattr(bot, '_replay_tags_sent'):
        bot._replay_tags_sent = set()
    
    tags = []
    
    # Tag rush type when detected (Zerg only, once per game)
    # Labels: 12_pool, speedling, macro (macro = not a rush)
    if (bot.enemy_race in {Race.Zerg, Race.Random} 
        and hasattr(bot, '_ling_rushed_v2') 
        and bot._ling_rushed_v2
        and hasattr(bot, '_rush_label')
        and bot._rush_label in {'12_pool', 'speedling'}
        and 'Rush' not in bot._replay_tags_sent):
        
        tags.append(f"Rush_{bot._rush_label}")
        bot._replay_tags_sent.add('Rush')
    
    # Tag worker rush
    if not bot._not_worker_rush and 'WorkerRush' not in bot._replay_tags_sent:
        tags.append("WorkerRush")
        bot._replay_tags_sent.add('WorkerRush')
    
    # Tag cannon rush
    if bot._cannon_rush_response and 'CannonRush' not in bot._replay_tags_sent:
        tags.append("CannonRush")
        bot._replay_tags_sent.add('CannonRush')
    
    # Tag marine rush
    if bot.mediator.get_enemy_marine_rush and 'MarineRush' not in bot._replay_tags_sent:
        tags.append("MarineRush")
        bot._replay_tags_sent.add('MarineRush')
    
    # Tag marauder rush
    if bot.mediator.get_enemy_marauder_rush and 'MarauderRush' not in bot._replay_tags_sent:
        tags.append("MarauderRush")
        bot._replay_tags_sent.add('MarauderRush')
    
    # Tag proxy zealot
    if bot.mediator.get_is_proxy_zealot and 'ProxyZealot' not in bot._replay_tags_sent:
        tags.append("ProxyZealot")
        bot._replay_tags_sent.add('ProxyZealot')
    
    # Tag four gate
    if bot.mediator.get_enemy_four_gate and 'FourGate' not in bot._replay_tags_sent:
        tags.append("FourGate")
        bot._replay_tags_sent.add('FourGate')
    
    # Tag roach rush
    if bot.mediator.get_enemy_roach_rushed and 'RoachRush' not in bot._replay_tags_sent:
        tags.append("RoachRush")
        bot._replay_tags_sent.add('RoachRush')
    
    # Tag ravager rush
    if bot.mediator.get_enemy_ravager_rush and 'RavagerRush' not in bot._replay_tags_sent:
        tags.append("RavagerRush")
        bot._replay_tags_sent.add('RavagerRush')
    
    return tags


def _print_army_composition(bot) -> None:
    """Print breakdown of army unit types and totals."""
    if not bot.own_army:
        print("    No army units")
        return
    
    # Count units by type
    unit_counts = {}
    for unit in bot.own_army:
        type_name = unit.type_id.name
        unit_counts[type_name] = unit_counts.get(type_name, 0) + 1
    
    # Print sorted by count (descending)
    for unit_type, count in sorted(unit_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {unit_type}: {count}")
    
    print(f"    Total Army: {len(bot.own_army)} units")


def _print_unit_roles(bot) -> None:
    """Print unit role assignments and squad counts."""
    try:
        attacking = bot.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        defending = bot.mediator.get_units_from_role(role=UnitRole.DEFENDING)
        base_defenders = bot.mediator.get_units_from_role(role=UnitRole.BASE_DEFENDER)
        
        print(f"    ATTACKING: {len(attacking)}")
        print(f"    DEFENDING: {len(defending)}")
        
        # Show defender unit composition if there are defenders
        if defending:
            defender_counts = {}
            for unit in defending:
                type_name = unit.type_id.name
                defender_counts[type_name] = defender_counts.get(type_name, 0) + 1
            
            defender_list = ", ".join([f"{count}x{unit_type}" for unit_type, count in sorted(defender_counts.items())])
            print(f"      └─ Defenders: {defender_list}")
        
        print(f"    BASE_DEFENDER: {len(base_defenders)}")
        
        # Squad counts
        from bot.constants import ATTACKING_SQUAD_RADIUS, DEFENDER_SQUAD_RADIUS
        attacking_squads = bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
        defending_squads = bot.mediator.get_squads(role=UnitRole.DEFENDING, squad_radius=ATTACKING_SQUAD_RADIUS)
        base_squads = bot.mediator.get_squads(role=UnitRole.BASE_DEFENDER, squad_radius=DEFENDER_SQUAD_RADIUS)
        
        print(f"    Squads: ATK:{len(attacking_squads)} DEF:{len(defending_squads)} BASE:{len(base_squads)}")
    except Exception as e:
        print(f"    Error getting roles: {e}")


def _print_enemy_intel(bot) -> None:
    """Print scouted enemy units and structures."""
    if not bot.enemy_units and not bot.enemy_structures:
        print("    No enemy scouted")
        return
    
    # Count enemy units
    if bot.enemy_units:
        unit_counts = {}
        for unit in bot.enemy_units:
            type_name = unit.type_id.name
            unit_counts[type_name] = unit_counts.get(type_name, 0) + 1
        
        print("    Enemy Units:")
        for unit_type, count in sorted(unit_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {unit_type}: {count}")
    
    # Count enemy structures
    if bot.enemy_structures:
        structure_counts = {}
        for structure in bot.enemy_structures:
            type_name = structure.type_id.name
            structure_counts[type_name] = structure_counts.get(type_name, 0) + 1
        
        print("    Enemy Structures:")
        for struct_type, count in sorted(structure_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {struct_type}: {count}")


def _print_economy_state(bot) -> None:
    """Print economy state (bases, workers, resources)."""
    gatherers = bot.mediator.get_units_from_role(role=UnitRole.GATHERING)
    economy_state = get_economy_state(bot)
    
    print(f"    State: {economy_state}")
    print(f"    Bases: {len(bot.townhalls)}")
    print(f"    Workers: {len(bot.workers)} (Gathering: {len(gatherers)})")
    print(f"    Supply: {bot.supply_used}/{bot.supply_cap}")
    print(f"    Minerals: {bot.minerals}")
    print(f"    Vespene: {bot.vespene}")
    print(f"    Income: {bot.state.score.collection_rate_minerals}/min minerals, {bot.state.score.collection_rate_vespene}/min gas")


def _print_rush_detection_status(bot) -> None:
    """Print rush detection intel (Zerg only - early game)."""
    if not hasattr(bot, '_rush_label') or bot.time > 240.0:
        return
    
    print("\n  RUSH DETECTION (vs Zerg):")
    
    # Rush scores and classification
    score_12p = getattr(bot, '_score_12p', 0)
    score_speed = getattr(bot, '_score_speed', 0)
    rush_label = getattr(bot, '_rush_label', 'none')
    is_rushed = getattr(bot, '_ling_rushed_v2', False)
    auto_true = getattr(bot, '_auto_true_fired', False)
    rush_source = getattr(bot, '_rush_source', None)
    
    # ML probabilities (if available)
    ml_probs = getattr(bot, '_ml_probs', None)
    ml_confidence = getattr(bot, '_ml_confidence', None)
    
    source_str = f" [{rush_source}]" if rush_source else ""
    print(f"    Label: {rush_label}{source_str} (12p={score_12p}, speed={score_speed})")
    
    if ml_probs:
        prob_str = ", ".join(f"{k}={v*100:.0f}%" for k, v in sorted(ml_probs.items()))
        print(f"    ML Probs: {prob_str}")
    
    print(f"    Rush Detected: {is_rushed} (auto-TRUE={auto_true})")
    
    # Natural scouting
    natural_scouted = getattr(bot, '_natural_ever_scouted', False)
    nat_started = getattr(bot, '_enemy_nat_started_at', None)
    nat_str = f"{nat_started:.1f}s" if nat_started else "absent"
    print(f"    Enemy Natural: {nat_str} (scouted={natural_scouted})")
    
    # Pool info
    pool_state = getattr(bot, '_pool_seen_state', 'none')
    pool_time = getattr(bot, '_pool_seen_time', None)
    pool_time_str = f"{pool_time:.1f}s" if pool_time else "unknown"
    print(f"    Pool: {pool_state} (start≈{pool_time_str})")
    
    # Speed research
    speed_started = getattr(bot, '_speed_research_started', False)
    speed_time = getattr(bot, '_speed_research_time', None)
    speed_str = f"{speed_time:.1f}s" if speed_time else "not seen"
    if speed_started:
        print(f"    Speed Research: started at {speed_str}")


def print_end_game_report(
    performance_monitor,
    game_result,
    game_time: float,
    idle_worker_time: float,
    idle_production_time: float
) -> None:
    """
    Print streamlined end-game report using PerformanceMonitor data.
    
    Args:
        performance_monitor: PerformanceMonitor instance with collected data
        game_result: Result enum (Victory/Defeat/Tie)
        game_time: Game time in seconds
        idle_worker_time: Total idle worker time in seconds
        idle_production_time: Total idle production time in seconds
    """
    # Get combined SQ score from monitor
    sq = performance_monitor.get_current_sq()
    rating = performance_monitor.get_efficiency_rating()
    
    # Calculate combined metrics for display
    combined_income = performance_monitor.avg_income_minerals + performance_monitor.avg_income_vespene
    combined_unspent = performance_monitor.avg_unspent_minerals + performance_monitor.avg_unspent_vespene
    
    # Print clean report
    print("\n" + "="*50)
    print(f"  RESULT: {game_result}")
    print(f"  Game Time: {game_time / 60:.1f} min")
    print("="*50)
    print(f"  SPENDING QUOTIENT:  {sq:.1f}  [{rating}]")
    print("-"*50)
    print("  RESOURCES:")
    print(f"    Combined Income:  {combined_income:.0f}/min")
    print(f"    Combined Unspent: {combined_unspent:.0f}")
    print(f"      └─ Minerals: {performance_monitor.avg_unspent_minerals:.0f}m")
    print(f"      └─ Gas:      {performance_monitor.avg_unspent_vespene:.0f}g")
    print("-"*50)
    print("  EFFICIENCY:")
    print(f"    Idle Workers:     {idle_worker_time:.1f}s")
    print(f"    Idle Production:  {idle_production_time:.1f}s")
    print("="*50 + "\n")
