"""
Rush detection utilities for identifying early Zerg aggression.

Purpose: Classify Zerg opener as 12_pool, speedling, or none (hybrid rule + ML system)
Key Decisions: Pool timing is primary discriminator; auto-TRUE guards for clear rushes
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
    
    # Get ground distance to enemy position using new lazy evaluation API
    path = dijkstra_result.get_path(enemy_pos)
    ground_distance = len(path) if path else 0
    
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
    Three-label zergling rush detection with Auto-TRUE guards.
    
    Labels:
        - 12_pool: Very early pool (~12 supply), earliest ling rush
        - speedling: Early pool + gas for fast Metabolic Boost timing
        - none: Not a rush (negative class)
    
    Returns:
        True if rush detected (12_pool or speedling), False if none
    """
    # Initialize tracking attributes
    if not hasattr(bot, '_ling_rushed_v2'):
        bot._ling_rushed_v2 = False
        bot._rush_label = "none"
        bot._score_12p = 0
        bot._score_speed = 0
        bot._auto_true_fired = False
    
    # Return early if already detected as rush
    if bot._ling_rushed_v2:
        return True
    
    # Update signals (triggers _track_enemy_timings)
    signals = get_ling_rush_signals(bot)
    
    # === TIMING CONSTANTS ===
    T_POOL_12P_START_MIN = 38.0   # 0:38
    T_POOL_12P_START_MAX = 42.0   # 0:42
    T_POOL_12P_DONE = 65.0        # 1:05
    T_POOL_SPEED_START_MIN = 48.0 # 0:48
    T_POOL_SPEED_START_MAX = 52.0 # 0:52
    T_NAT_CHECK = 80.0            # 1:20
    T_LING_EARLY = 105.0          # 1:45 (auto-TRUE)
    T_LING_12P_SEEN = 115.0       # 1:55
    T_CONTACT_SLOW = 120.0        # 2:00 (auto-TRUE slow-ling)
    T_CONTACT_SPEED = 160.0       # 2:40 (auto-TRUE speed-ling)
    T_SPEED_START_EARLY = 85.0    # 1:25
    T_QUEEN_CHECK = 120.0         # 2:00
    
    time_now = bot.time
    
    # === AUTO-TRUE GUARDS (evaluate first) ===
    
    # Guard A: Any ling seen ≤ 1:45 → 12_pool
    if bot._first_ling_seen_time is not None and bot._first_ling_seen_time <= T_LING_EARLY:
        bot._ling_rushed_v2 = True
        bot._rush_label = "12_pool"
        bot._auto_true_fired = True
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE A): Ling seen at {bot._first_ling_seen_time:.1f}s → 12_pool")
        return True
    
    # Guard B: Slow-ling contact ≤ 2:00 → 12_pool
    if (bot._first_ling_contact_nat_time is not None and 
        bot._first_ling_contact_nat_time <= T_CONTACT_SLOW and
        not bot._ling_has_speed):
        bot._ling_rushed_v2 = True
        bot._rush_label = "12_pool"
        bot._auto_true_fired = True
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE B): Slow-ling contact at {bot._first_ling_contact_nat_time:.1f}s → 12_pool")
        return True
    
    # Guard C: Speed-ling contact ≤ 2:40 → speedling
    if (bot._first_ling_contact_nat_time is not None and 
        bot._first_ling_contact_nat_time <= T_CONTACT_SPEED and
        bot._ling_has_speed):
        bot._ling_rushed_v2 = True
        bot._rush_label = "speedling"
        bot._auto_true_fired = True
        print(f"{bot.time_formatted}: Rush detected (Auto-TRUE C): Speed-ling contact at {bot._first_ling_contact_nat_time:.1f}s → speedling")
        return True
    
    # === SCORING ===
    
    score_12p = 0   # 12-pool score
    score_speed = 0 # Speedling score
    
    # Get pool start time estimate
    pool_start = bot._pool_seen_time  # Already estimated start time
    
    # === 12_POOL SIGNALS ===
    
    # Early pool start (0:38-0:42)
    if pool_start is not None and T_POOL_12P_START_MIN <= pool_start <= T_POOL_12P_START_MAX:
        score_12p += 4
    
    # Pool done early (≤1:05)
    if bot._pool_seen_state == "done" and pool_start is not None and pool_start <= T_POOL_12P_START_MAX:
        score_12p += 3
    
    # No natural by 1:20
    no_natural = bot._enemy_nat_started_at is None and time_now >= T_NAT_CHECK
    if no_natural:
        score_12p += 3
    
    # Very early lings (≤1:55)
    if bot._first_ling_seen_time is not None and bot._first_ling_seen_time <= T_LING_12P_SEEN:
        score_12p += 4
    
    # No queen by 2:00
    no_queen = (bot._pool_seen_state in {"morphing", "done"} and 
                bot._queen_started_time is None and 
                time_now >= T_QUEEN_CHECK)
    if no_queen:
        score_12p += 2
    
    # === SPEEDLING SIGNALS ===
    
    # Speedling pool timing (0:48-0:52)
    if pool_start is not None and T_POOL_SPEED_START_MIN <= pool_start <= T_POOL_SPEED_START_MAX:
        score_speed += 3
    
    # Early gas with workers mining
    if bot._extractor_seen_time is not None and bot._gas_workers_count >= 2:
        score_speed += 3
    
    # Speed research early (≤1:25)
    if bot._speed_research_started and bot._speed_research_time is not None:
        if bot._speed_research_time <= T_SPEED_START_EARLY:
            score_speed += 4
    
    # Slow lings appear (2:05-2:15) - indicates speedling build before speed finishes
    if bot._first_ling_seen_time is not None and 125.0 <= bot._first_ling_seen_time <= 135.0:
        score_speed += 2
    
    # Speed-ling contact ≤2:40
    if (bot._first_ling_contact_nat_time is not None and 
        bot._first_ling_contact_nat_time <= T_CONTACT_SPEED and 
        bot._ling_has_speed):
        score_speed += 4
    
    # Dampers (reduce speedling score if looks macro)
    # Natural on time
    if bot._enemy_nat_started_at is not None and bot._enemy_nat_started_at <= T_NAT_CHECK:
        score_speed -= 1
    
    # Normal queen timing
    if bot._queen_started_time is not None and 110.0 <= bot._queen_started_time <= 120.0:
        score_speed -= 1
    
    # Update bot state
    bot._score_12p = score_12p
    bot._score_speed = score_speed
    
    # === CLASSIFICATION ===
    
    # 12_pool has priority if both score high
    if score_12p >= 5:
        bot._ling_rushed_v2 = True
        bot._rush_label = "12_pool"
        print(f"{bot.time_formatted}: Rush detected! 12_pool (score={score_12p})")
        return True
    
    if score_speed >= 5:
        bot._ling_rushed_v2 = True
        bot._rush_label = "speedling"
        print(f"{bot.time_formatted}: Rush detected! speedling (score={score_speed})")
        return True
    
    # Default: none (not a rush)
    bot._rush_label = "none"
    return False


def log_rush_detection_result(bot: "PiG_Bot", game_result):
    """
    Log rush detection results at game end for ML training and analysis.
    
    Creates a JSON log entry with all timing data and the final outcome.
    Appends to data/rush_detection_log.jsonl (one line per game).
    Feature names match the ML spec for direct use in training.
    
    Call this from bot.on_end() method.
    
    Args:
        bot: PiG_Bot instance
        game_result: Result enum from on_end parameter
    """
    # Skip if we haven't initialized tracking yet
    if not hasattr(bot, '_rush_label'):
        return
    
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "rush_detection_log.jsonl"
    
    # Get feature values, using -1 for missing (ML convention)
    def get_time(attr, default=-1):
        val = getattr(bot, attr, None)
        return val if val is not None else default
    
    log_entry = {
        # Metadata
        "map_name": bot.game_info.map_name,
        "enemy_race": str(bot.enemy_race),
        "rush_distance_seconds": getattr(bot, '_rush_time_seconds', -1),
        
        # Raw timing features (for ML) - use -1 for missing
        "pool_start": get_time('_pool_seen_time'),
        "nat_start": get_time('_enemy_nat_started_at'),
        "gas_time": get_time('_extractor_seen_time'),
        "queen_time": get_time('_queen_started_time'),
        "ling_seen": get_time('_first_ling_seen_time'),
        "ling_contact": get_time('_first_ling_contact_nat_time'),
        "speed_start": get_time('_speed_research_time'),
        "ling_has_speed": 1 if getattr(bot, '_ling_has_speed', False) else 0,
        "gas_workers": getattr(bot, '_gas_workers_count', 0),
        
        # Rule scores (for ML)
        "score_12p": getattr(bot, '_score_12p', 0),
        "score_speed": getattr(bot, '_score_speed', 0),
        
        # Classification result
        "auto_true_fired": getattr(bot, '_auto_true_fired', False),
        "rush_label": getattr(bot, '_rush_label', "none"),
        
        # Game outcome (for training labels / analysis)
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
