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
    Compute rush distance for logging purposes only.
    
    Uses Dijkstra pathfinding from our start location to enemy start location,
    offset by 3 tiles to avoid blocked building positions.
    
    Stores rush_time_seconds on bot object for logging.
    Returns tier string for backward compatibility.
    
    NOTE: Detection logic uses FIXED timing constants, not map-aware offsets.
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
    
    # Calculate rush time for logging only
    rush_time = (ground_distance / RUSH_SPEED) * RUSH_DISTANCE_CALIBRATION
    
    # Store for logging
    bot._rush_time_seconds = rush_time
    
    # Return tier for backward compatibility
    if rush_time <= 36:
        tier = "short"
    elif rush_time <= 45:
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
    if not hasattr(bot, '_gas_workers_count'):
        bot._gas_workers_count = 0
    if not hasattr(bot, '_speed_research_started'):
        bot._speed_research_started = False
    if not hasattr(bot, '_speed_research_time'):
        bot._speed_research_time = None
    if not hasattr(bot, '_baneling_nest_seen_time'):
        bot._baneling_nest_seen_time = None
    if not hasattr(bot, '_ling_has_speed'):
        bot._ling_has_speed = False
    
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
        
        # Check if lings have speed (movement speed check)
        if not bot._ling_has_speed:
            zergling_base_speed = bot._game_data.units[UnitTypeId.ZERGLING.value]._proto.movement_speed
            if any(ling.movement_speed > zergling_base_speed + 0.1 for ling in enemy_lings):
                bot._ling_has_speed = True
    
    # Track gas workers (count workers near enemy extractors)
    extractors = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.EXTRACTOR and s.is_ready]
    if extractors:
        workers_on_gas = [
            w for w in bot.enemy_units 
            if w.type_id == UnitTypeId.DRONE and w.distance_to(extractors[0]) < 3
        ]
        bot._gas_workers_count = len(workers_on_gas)
    
    # Track speed research (Metabolic Boost upgrade in progress)
    if not bot._speed_research_started and pools:
        pool = pools[0]
        # Check if pool is researching (has orders) - Metabolic Boost
        if pool.is_ready and pool.orders:
            bot._speed_research_started = True
            bot._speed_research_time = bot.time
    
    # Track baneling nest
    if bot._baneling_nest_seen_time is None:
        nests = [s for s in bot.enemy_structures if s.type_id == UnitTypeId.BANELINGNEST]
        if nests:
            bot._baneling_nest_seen_time = bot.time


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
    Three-type zergling rush detection with Auto-TRUE guards.
    
    Detects R1 (12p gasless), R2 (12p 11gas), R3 (13/12 speedling).
    Uses fixed timing constants (not map-aware).
    
    Returns:
        True if rush detected (Auto-TRUE or score >= 5), False otherwise
    """
    # Initialize tracking attributes
    if not hasattr(bot, '_ling_rushed_v2'):
        bot._ling_rushed_v2 = False
        bot._rush_score = 0
        bot._rush_label = "none"
        bot._r1_score = 0
        bot._r2_score = 0
        bot._r3_score = 0
    
    # Return early if already detected
    if bot._ling_rushed_v2:
        return True
    
    # Update signals (triggers _track_enemy_timings)
    signals = get_ling_rush_signals(bot)
    
    # Fixed timing constants 
    T_POOL_CHECK = 65.0
    T_NAT_CHECK = 80.0
    T_LING_EARLY = 105.0
    T_LING_SOFT = 110.0
    T_CONTACT_SLOW = 120.0
    T_CONTACT_FAST = 150.0
    T_QUEEN_CHECK = 125.0
    
    time_now = bot.time
    
    # === AUTO-TRUE GUARDS (evaluate first) ===
    
    # Guard A: Any ling seen ≤ 1:45
    if bot._first_ling_seen_time is not None and bot._first_ling_seen_time <= T_LING_EARLY:
        bot._ling_rushed_v2 = True
        bot._rush_label = "auto_true_early_ling"
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE A): Ling seen at {bot._first_ling_seen_time:.1f}s")
        return True
    
    # Guard B: Slow-ling contact ≤ 2:00
    if (bot._first_ling_contact_nat_time is not None and 
        bot._first_ling_contact_nat_time <= T_CONTACT_SLOW and
        not bot._ling_has_speed):
        bot._ling_rushed_v2 = True
        bot._rush_label = "auto_true_slow_contact"
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE B): Slow-ling contact at {bot._first_ling_contact_nat_time:.1f}s")
        return True
    
    # Guard C: Speed-ling contact ≤ 2:30
    if (bot._first_ling_contact_nat_time is not None and 
        bot._first_ling_contact_nat_time <= T_CONTACT_FAST and
        bot._ling_has_speed):
        bot._ling_rushed_v2 = True
        bot._rush_label = "auto_true_speed_contact"
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE C): Speed-ling contact at {bot._first_ling_contact_nat_time:.1f}s")
        return True
    
    # === SCORING: Three Rush Types ===
    
    r1_score = 0  # 12-Pool Gasless
    r2_score = 0  # 12-Pool 11-Gas  
    r3_score = 0  # 13/12 Speedling
    
    # Common features
    no_natural = bot._enemy_nat_started_at is None and time_now >= T_NAT_CHECK
    pool_early = (bot._pool_seen_state in {"morphing", "done"} and 
                  bot._pool_seen_time is not None and 
                  bot._pool_seen_time <= T_POOL_CHECK)
    early_lings = bot._first_ling_seen_time is not None and bot._first_ling_seen_time <= T_LING_SOFT
    no_gas = bot._extractor_seen_time is None and time_now >= T_NAT_CHECK
    no_queen = (bot._pool_seen_state in {"morphing", "done"} and 
                bot._queen_started_time is None and 
                time_now >= T_QUEEN_CHECK)
    
    # === R1: 12-Pool Gasless ===
    if no_natural:
        r1_score += 3
    if pool_early and no_natural:
        r1_score += 3
    if no_gas:
        r1_score += 2
    if no_queen:
        r1_score += 1
    if early_lings:
        r1_score += 4
    
    # === R2: 12-Pool 11-Gas (ling-bane or early speed) ===
    if no_natural:
        r2_score += 3
    if pool_early:
        r2_score += 3
    # Active gas: extractor + workers on gas
    if bot._extractor_seen_time is not None and bot._gas_workers_count >= 2 and time_now <= 70.0:
        r2_score += 2
    # Speed research
    if bot._speed_research_started and bot._speed_research_time is not None and bot._speed_research_time <= 110.0:
        r2_score += 2
    if early_lings:
        r2_score += 3
    # Baneling nest
    if bot._baneling_nest_seen_time is not None and 120.0 <= bot._baneling_nest_seen_time <= 140.0:
        r2_score += 2
    if no_queen:
        r2_score += 1
    
    # === R3: 13/12 Speedling ===
    # 13 pool timing (starts later)
    pool_13 = (bot._pool_seen_time is not None and 
               35.0 <= bot._pool_seen_time <= 40.0 and
               bot._extractor_seen_time is not None and bot._gas_workers_count >= 2)
    if pool_13:
        r3_score += 3
    if no_natural:
        r3_score += 2
    if bot._speed_research_started and bot._speed_research_time is not None and bot._speed_research_time <= 110.0:
        r3_score += 2
    # Speed contact
    if (bot._first_ling_contact_nat_time is not None and 
        bot._first_ling_contact_nat_time <= T_CONTACT_FAST and 
        bot._ling_has_speed):
        r3_score += 4
    if early_lings:
        r3_score += 3
    if no_queen:
        r3_score += 1
    
    # === Aggregate Score ===
    total_score = r1_score + r2_score + r3_score
    
    # Determine highest-scoring rush type
    if max(r1_score, r2_score, r3_score) == r1_score and r1_score > 0:
        rush_label = "12p_gasless"
    elif max(r1_score, r2_score, r3_score) == r2_score and r2_score > 0:
        rush_label = "12p_11gas"
    elif max(r1_score, r2_score, r3_score) == r3_score and r3_score > 0:
        rush_label = "13_12_speed"
    else:
        rush_label = "none"
    
    # Update bot state
    bot._rush_score = total_score
    bot._r1_score = r1_score
    bot._r2_score = r2_score
    bot._r3_score = r3_score
    bot._rush_label = rush_label
    
    # Threshold: score >= 5 triggers rush response
    rush_detected = total_score >= 5
    
    # Trigger detection
    if rush_detected:
        bot._ling_rushed_v2 = True
        print(f"{bot.time_formatted}: Rush detected! Score={total_score} (R1={r1_score}, R2={r2_score}, R3={r3_score}), Type={rush_label}")
    
    return bot._ling_rushed_v2


def log_rush_detection_result(bot: "PiG_Bot", game_result):
    """
    Log rush detection results at game end for ML training and analysis.
    
    Creates a JSON log entry with all timing data and the final outcome.
    Appends to data/rush_detection_log.jsonl (one line per game).
    
    Call this from bot.on_end() method.
    
    Args:
        bot: PiG_Bot instance
        game_result: Result enum from on_end parameter
    """
    # Skip if we haven't initialized tracking yet
    if not hasattr(bot, '_rush_score'):
        return
    
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "rush_detection_log.jsonl"
    
    log_entry = {
        "map_name": bot.game_info.map_name,
        "enemy_race": str(bot.enemy_race),
        "rush_distance_seconds": getattr(bot, '_rush_time_seconds', None),
        
        # Timing observations
        "t_nat_started": getattr(bot, '_enemy_nat_started_at', None),
        "pool_seen_state": getattr(bot, '_pool_seen_state', "none"),
        "t_pool_seen": getattr(bot, '_pool_seen_time', None),
        "t_extractor_seen": getattr(bot, '_extractor_seen_time', None),
        "t_queen_started": getattr(bot, '_queen_started_time', None),
        "t_first_ling_seen": getattr(bot, '_first_ling_seen_time', None),
        "t_first_ling_contact_nat": getattr(bot, '_first_ling_contact_nat_time', None),
        "ling_has_speed": getattr(bot, '_ling_has_speed', False),
        
        # Detection scores
        "total_score": getattr(bot, '_rush_score', 0),
        "r1_score": getattr(bot, '_r1_score', 0),
        "r2_score": getattr(bot, '_r2_score', 0),
        "r3_score": getattr(bot, '_r3_score', 0),
        "rush_label": getattr(bot, '_rush_label', "none"),
        "is_rushed": getattr(bot, '_ling_rushed_v2', False),
        
        # Game outcome
        "result": str(game_result),
        "game_time_seconds": bot.time,
    }
    
    # Append to JSONL (one line per game)
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        print(f"Rush detection logged to {log_file}")
    except Exception as e:
        print(f"Failed to log rush detection: {e}")
