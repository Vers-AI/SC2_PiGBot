"""
Rush detection utilities for identifying early Zerg aggression.

Purpose: Compute rush distance tiers and evaluate ling rush signals
Key Decisions: Use map_data.pathfind for ground distance, dynamic zergling speed from game data
Limitations: Requires enemy main location scouted; probe death reduces signal quality
"""

from typing import TYPE_CHECKING
import json
from pathlib import Path

import numpy as np
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole

from cython_extensions import cy_dijkstra
from bot.constants import RUSH_SPEED, RUSH_DISTANCE_CALIBRATION


if TYPE_CHECKING:
    from bot.bot import PiG_Bot


def compute_rush_distance_tier(bot: "PiG_Bot") -> str:
    """
    Compute map offset for rush timing based on zergling travel time.
    
    Uses Dijkstra pathfinding from our start location to enemy start location,
    offset by 3 tiles to avoid blocked building positions.
    
    Stores map_offset and rush_time_zergling_seconds on bot object.
    Returns tier string for backward compatibility (deprecated).
    """
    # Offset start locations by 3 tiles to get off the Nexus/Hatchery
    our_offset_pos = bot.start_location.towards(bot.enemy_start_locations[0], 3)
    enemy_pos = bot.enemy_start_locations[0]
    
    our_x, our_y = int(our_offset_pos.x), int(our_offset_pos.y)
    enemy_x, enemy_y = int(enemy_pos.x), int(enemy_pos.y)
    
    # Create cost grid from SC2 pathing grid 
    cost_grid = np.where(
        bot.game_info.pathing_grid.data_numpy.T == 1, 
        1.0, 
        np.inf
    ).astype(np.float64)
    
    # Run Dijkstra from our position 
    targets = np.array([[our_x, our_y]], dtype=np.intp)
    dijkstra_result = cy_dijkstra(cost_grid, targets, checks_enabled=True)
    
    # Get ground distance to enemy position
    ground_distance = dijkstra_result.distance[enemy_x, enemy_y]
    
    if ground_distance <= 0 or ground_distance == float('inf'):
        # Pathfind failed, default to medium tier
        bot._rush_time_seconds = 0.0  # Mark as unknown
        return "medium"
    
    # Calculate rush time using worker speed (our current RUSH_SPEED)
    rush_time_worker = (ground_distance / RUSH_SPEED) * RUSH_DISTANCE_CALIBRATION
    
    # Convert to zergling time (zergling = 4.13 tiles/s, worker = 3.94 tiles/s)
    # Zerglings are slower, so rush time is LONGER
    rush_time_zergling = rush_time_worker * (RUSH_SPEED / 4.13)
    
    # Compute map offset relative to 120s reference (medium map)
    # Negative on short maps, positive on long maps
    map_offset = rush_time_zergling - 120.0
    
    # Store both values on bot
    bot._rush_time_seconds = rush_time_worker  # Keep for backward compat
    bot._rush_time_zergling_seconds = rush_time_zergling
    bot._map_offset = map_offset
    
    # Derive map-aware checkpoints (computed once, used throughout detection)
    bot._T_POOL_CHECK = 65.0 + map_offset
    bot._T_NAT_CHECK = 80.0 + map_offset
    bot._T_LING_SIGHT = 105.0 + map_offset
    bot._T_LING_CONTACT_NAT = 120.0 + map_offset
    bot._T_QUEEN_CHECK = 125.0 + map_offset
    
    # Return tier for backward compatibility (deprecated, use checkpoints instead)
    if rush_time_zergling <= 110:
        tier = "short"
    elif rush_time_zergling <= 130:
        tier = "medium"
    else:
        tier = "long"
    
    return tier


def _estimate_building_start_time(bot: "PiG_Bot", building) -> float:
    """
    Estimate when a building started based on current build progress.
    
    Formula from spec:
        start_time = current_time - (build_progress * build_duration / 22.4)
    
    Args:
        building: Unit object with build_progress and _type_data
    
    Returns:
        Estimated game time when building started morphing
    """
    build_duration = building._type_data.cost.time  # In game loops
    build_progress = building.build_progress  # 0.0 to 1.0
    
    # Convert game loops to Faster-speed seconds (22.4 loops/second)
    elapsed_build_time = (build_progress * build_duration) / 22.4
    
    return bot.time - elapsed_build_time


def _track_enemy_timings(bot: "PiG_Bot"):
    """
    Track enemy building and unit timings for rush detection.
    
    Stores timestamps on bot object:
        - _enemy_nat_started_at: float | None
        - _pool_seen_state: str (none/morphing/done/unknown)
        - _pool_seen_time: float | None
        - _extractor_seen_time: float | None
        - _queen_started_time: float | None
        - _first_ling_seen_time: float | None
        - _first_ling_contact_nat_time: float | None
    """
    # Initialize tracking attributes
    if not hasattr(bot, '_enemy_nat_started_at'):
        bot._enemy_nat_started_at = None
    if not hasattr(bot, '_pool_seen_state'):
        bot._pool_seen_state = "none"
    if not hasattr(bot, '_pool_seen_time'):
        bot._pool_seen_time = None
    if not hasattr(bot, '_extractor_seen_time'):
        bot._extractor_seen_time = None
    if not hasattr(bot, '_queen_started_time'):
        bot._queen_started_time = None
    if not hasattr(bot, '_first_ling_seen_time'):
        bot._first_ling_seen_time = None
    if not hasattr(bot, '_first_ling_contact_nat_time'):
        bot._first_ling_contact_nat_time = None
    
    enemy_nat_pos = bot.mediator.get_enemy_nat
    
    # Track natural hatchery
    if bot._enemy_nat_started_at is None:
        nat_hatcheries = [
            s for s in bot.enemy_structures
            if s.type_id == UnitTypeId.HATCHERY and s.distance_to(enemy_nat_pos) < 10
        ]
        if nat_hatcheries:
            hatch = nat_hatcheries[0]
            if hatch.is_ready:
                # Estimate backwards from completion
                bot._enemy_nat_started_at = _estimate_building_start_time(bot, hatch)
            else:
                # Estimate from current progress
                bot._enemy_nat_started_at = _estimate_building_start_time(bot, hatch)
    
    # Track spawning pool
    pools = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.SPAWNINGPOOL]
    if pools and bot._pool_seen_state == "none":
        pool = pools[0]
        bot._pool_seen_time = _estimate_building_start_time(bot, pool)
        if pool.is_ready:
            bot._pool_seen_state = "done"
        else:
            bot._pool_seen_state = "morphing"
    elif pools and bot._pool_seen_state == "morphing":
        # Update to done if completed
        if pools[0].is_ready:
            bot._pool_seen_state = "done"
    
    # Track extractor (first one)
    if bot._extractor_seen_time is None:
        extractors = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.EXTRACTOR]
        if extractors:
            bot._extractor_seen_time = _estimate_building_start_time(bot, extractors[0])
    
    # Track queen (first one)
    if bot._queen_started_time is None:
        queens = [u for u in bot.enemy_units if u.type_id == UnitTypeId.QUEEN]
        if queens:
            # Queens can't estimate start time easily, use first seen time
            bot._queen_started_time = bot.time
    
    # Track zerglings
    enemy_lings = bot.mediator.get_enemy_army_dict.get(UnitTypeId.ZERGLING, [])
    
    if enemy_lings:
        # First ling seen anywhere
        if bot._first_ling_seen_time is None:
            bot._first_ling_seen_time = bot.time
        
        # First ling contact at our natural
        if bot._first_ling_contact_nat_time is None:
            our_nat_pos = bot.mediator.get_own_nat
            lings_at_nat = [
                ling for ling in enemy_lings
                if ling.distance_to(our_nat_pos) < 15
            ]
            if lings_at_nat:
                bot._first_ling_contact_nat_time = bot.time


def _probe_scout_status(bot: "PiG_Bot") -> dict:
    """
    Track probe scout status for adjusting heuristics.
    
    Returns:
        {
            'scout_active': bool,
            'scout_died_early': bool,  # died before 2:00 without seeing natural
            'saw_enemy_natural': bool
        }
    """
    # Check if we have any active BUILD_RUNNER_SCOUT units
    scout_units = bot.mediator.get_units_from_role(
        role=UnitRole.BUILD_RUNNER_SCOUT, 
        unit_type=bot.worker_type
    )
    
    scout_active = len(scout_units) > 0
    
    # Initialize tracking attributes if needed
    if not hasattr(bot, '_scout_died_early'):
        bot._scout_died_early = False
    if not hasattr(bot, '_scout_was_active'):
        bot._scout_was_active = False
    
    # If scout was active last frame but gone now, and time < 120s, mark died early
    if bot._scout_was_active and not scout_active:
        if bot.time < 120.0 and not hasattr(bot, '_saw_enemy_natural_before_scout_died'):
            bot._scout_died_early = True
    
    bot._scout_was_active = scout_active
    
    # Check if we've seen enemy natural (structure visible there)
    saw_enemy_natural = any(
        bot.enemy_structures.closer_than(15, bot.mediator.get_enemy_nat)
    )
    
    if saw_enemy_natural:
        bot._saw_enemy_natural_before_scout_died = True
    
    return {
        'scout_active': scout_active,
        'scout_died_early': bot._scout_died_early,
        'saw_enemy_natural': saw_enemy_natural
    }


def get_ling_rush_signals(bot: "PiG_Bot") -> dict:
    """
    Gather all observable signals for rush detection.
    
    Updates enemy timing trackers and returns current state.
    
    Returns:
        Dictionary with keys:
            - tier: str ('short', 'medium', 'long') - deprecated
            - natural_absent: bool
            - lings_near_base: int
            - total_lings: int
            - speed_seen: bool
            - scout_died_early: bool
            - saw_natural: bool
    """
    # Update enemy timings first
    _track_enemy_timings(bot)
    
    tier = bot.rush_distance_tier
    scout_status = _probe_scout_status(bot)
    
    # Check if enemy has a natural expansion
    # Use scout status to determine if we've seen the natural
    # If scout has been to natural and didn't see structures = natural_absent=True
    # If scout hasn't been there yet = natural_absent=False (don't know yet)
    enemy_nat_pos = bot.mediator.get_enemy_nat
    
    # Track if we've ever had vision of the natural
    if not hasattr(bot, '_natural_ever_scouted'):
        bot._natural_ever_scouted = False
    
    # Check visibility in a wider area (5 radius) around natural
    # Also check fog of war state (1 = fogged/previously seen, 2 = currently visible)
    visibility_state = bot.state.visibility[enemy_nat_pos.rounded]
    currently_visible = visibility_state == 2
    previously_seen = visibility_state >= 1  # 1 = fogged (was seen), 2 = visible
    
    # Update if we have vision or if the area shows as previously seen
    if currently_visible or previously_seen:
        bot._natural_ever_scouted = True
    
    # Check for structures at natural (works with snapshots/memory too)
    structures_at_natural = bot.enemy_structures.closer_than(15, enemy_nat_pos)
    has_natural_structure = len(structures_at_natural) > 0
    
    # Natural is absent if: we've scouted it AND there's no structure there
    natural_absent = bot._natural_ever_scouted and not has_natural_structure
    
    # Debug: Print status (remove after testing)
    if bot.time < 180.0 and int(bot.time) % 30 == 0:
        print(f"DEBUG Natural Check: ever_scouted={bot._natural_ever_scouted}, has_structure={has_natural_structure}, absent={natural_absent}")
    
    # Ling counts
    enemy_lings = bot.mediator.get_enemy_army_dict.get(UnitTypeId.ZERGLING, [])
    lings_near_main = [
        ling for ling in enemy_lings 
        if ling.distance_to(bot.start_location) < 50
    ]
    
    # Speed detection - check for metabolic boost buff (BuffId is not easily accessible, use movement speed check)
    # If any ling is moving faster than base speed, speed is researched
    zergling_base_speed = bot._game_data.units[UnitTypeId.ZERGLING.value]._proto.movement_speed
    speed_seen = any(
        ling.movement_speed > zergling_base_speed + 0.1  # Small epsilon for floating point
        for ling in enemy_lings
    )
    
    return {
        'tier': tier,
        'natural_absent': natural_absent,
        'lings_near_base': len(lings_near_main),
        'total_lings': len(enemy_lings),
        'speed_seen': speed_seen,
        'scout_died_early': scout_status['scout_died_early'],
        'saw_natural': scout_status['saw_enemy_natural'],
    }


def get_enemy_ling_rushed_v2(bot: "PiG_Bot") -> bool:
    """
    Map-aware score-based ling rush detection (spec-compliant).
    
    Uses observable features scored against map-adjusted checkpoints:
    - Natural timing
    - Pool timing and state
    - Zergling sighting and contact timing
    - Gas and Queen timing (refiners)
    
    Returns:
        True if rush detected (score >= 5), False otherwise
        Debounced to prevent flapping
    """
    # Initialize tracking attributes
    if not hasattr(bot, '_ling_rushed_v2'):
        bot._ling_rushed_v2 = False
        bot._rush_score = 0
        bot._rush_label = "likely_macro"
        bot._rush_debounce_until = 0.0
    
    # Debounce: hold decision for 20s once triggered
    if bot._ling_rushed_v2 and bot.time < bot._rush_debounce_until:
        return True
    
    # Update signals (triggers _track_enemy_timings)
    signals = get_ling_rush_signals(bot)
    
    # Get map-aware checkpoints
    T_NAT_CHECK = bot._T_NAT_CHECK
    T_POOL_CHECK = bot._T_POOL_CHECK
    T_LING_SIGHT = bot._T_LING_SIGHT
    T_LING_CONTACT_NAT = bot._T_LING_CONTACT_NAT
    T_QUEEN_CHECK = bot._T_QUEEN_CHECK
    
    time_now = bot.time
    
    # === FEATURE EVALUATION ===
    
    # 1. No natural by T_NAT_CHECK (strong, 3 points)
    no_natural_by_T = (
        bot._enemy_nat_started_at is None and 
        time_now >= T_NAT_CHECK
    )
    
    # 2. Pool early with no natural (strong, 3 points)
    pool_early = (
        bot._pool_seen_state in {"morphing", "done"} and
        bot._pool_seen_time is not None and
        bot._pool_seen_time <= T_POOL_CHECK and
        bot._enemy_nat_started_at is None
    )
    
    # 3. Early lings seen (strong, 3 points)
    early_lings_seen = (
        bot._first_ling_seen_time is not None and
        bot._first_ling_seen_time <= T_LING_SIGHT
    )
    
    # 4. Ling contact by expected time (very strong, 4 points - confirmatory)
    ling_contact_by_T = (
        bot._first_ling_contact_nat_time is not None and
        bot._first_ling_contact_nat_time <= T_LING_CONTACT_NAT
    )
    
    # 5. Gasless early (refiner, 1 point)
    gasless_early = (
        bot._extractor_seen_time is None and
        time_now >= (80.0 + bot._map_offset)
    )
    
    # 6. Queen late with lings out (refiner, 1 point)
    queen_late_with_lings = (
        bot._queen_started_time is None and
        early_lings_seen and
        time_now >= T_QUEEN_CHECK
    )
    
    # === SCORING ===
    
    score = 0
    
    # Strong signals
    if no_natural_by_T:
        score += 3
    if pool_early:
        score += 3
    if early_lings_seen:
        score += 3
    if ling_contact_by_T:
        score += 4  # Confirmation
    
    # Refiners
    if gasless_early and early_lings_seen:
        score += 1
    if queen_late_with_lings:
        score += 1
    
    # False positive reducer: late natural with no lings
    if (bot._enemy_nat_started_at is not None and
        bot._enemy_nat_started_at <= T_NAT_CHECK + 15 and
        not early_lings_seen):
        score -= 1  # Defensive pool-first, not rush
    
    # === CLASSIFICATION ===
    
    if score >= 6:
        rush_label = "12pool_class_rush"
    elif 3 <= score <= 5:
        rush_label = "pool_first_pressure"
    else:
        rush_label = "likely_macro"
    
    # Update bot state
    bot._rush_score = score
    bot._rush_label = rush_label
    
    # Threshold: score >= 5 triggers rush response
    rush_detected = score >= 5
    
    # Debug output (remove after testing)
    if time_now < 240.0 and int(time_now) % 30 == 0:
        print(f"DEBUG Rush Score [{bot.time_formatted}]: score={score}, label={rush_label}, "
              f"no_nat={no_natural_by_T}, pool_early={pool_early}, lings_seen={early_lings_seen}, "
              f"contact={ling_contact_by_T}")
    
    # Trigger detection and set debounce
    if rush_detected and not bot._ling_rushed_v2:
        bot._ling_rushed_v2 = True
        bot._rush_debounce_until = time_now + 20.0  # Hold for 20 seconds
        print(f"{bot.time_formatted}: Ling rush detected! Score={score}, Label={rush_label}")
    
    # Downgrade rule: if natural appears + no contact by T+10, cancel alarm
    if (bot._ling_rushed_v2 and 
        time_now >= bot._rush_debounce_until and
        bot._enemy_nat_started_at is not None and
        (bot._first_ling_contact_nat_time is None or 
         bot._first_ling_contact_nat_time > T_LING_CONTACT_NAT + 10)):
        bot._ling_rushed_v2 = False
        bot._rush_label = "likely_macro"
        print(f"{bot.time_formatted}: Rush alarm downgraded - natural appeared, no ling contact")
    
    return bot._ling_rushed_v2


def log_rush_detection_result(bot: "PiG_Bot"):
    """
    Log rush detection results at game end for ML training and analysis.
    
    Creates a JSON log entry with all timing data and the final outcome.
    Appends to data/rush_detection_log.jsonl (one line per game).
    
    Call this from bot.on_end() method.
    """
    # Skip if we haven't initialized tracking yet
    if not hasattr(bot, '_map_offset'):
        return
    
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "rush_detection_log.jsonl"
    
    log_entry = {
        "map_name": bot.game_info.map_name,
        "enemy_race": str(bot.enemy_race),
        "rush_time_zergling_seconds": getattr(bot, '_rush_time_zergling_seconds', None),
        "map_offset": getattr(bot, '_map_offset', None),
        
        # Timing observations
        "t_nat_started": bot._enemy_nat_started_at,
        "pool_seen_state": bot._pool_seen_state,
        "t_pool_seen": bot._pool_seen_time,
        "t_extractor_seen": bot._extractor_seen_time,
        "t_queen_started": bot._queen_started_time,
        "t_first_ling_seen": bot._first_ling_seen_time,
        "t_first_ling_contact_nat": bot._first_ling_contact_nat_time,
        
        # Detection results
        "score": bot._rush_score,
        "rush_label": bot._rush_label,
        "rush_detected": bot._ling_rushed_v2,
        
        # Game outcome
        "result": str(bot.state.result) if hasattr(bot.state, 'result') else None,
        "game_time_seconds": bot.time,
    }
    
    # Append to JSONL (one line per game)
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        print(f"Rush detection logged to {log_file}")
    except Exception as e:
        print(f"Failed to log rush detection: {e}")
