"""
Rush detection utilities for identifying early Zerg aggression.

Purpose: Compute rush distance tiers and evaluate ling rush signals
Key Decisions: Use map_data.pathfind for ground distance, dynamic zergling speed from game data
Limitations: Requires enemy main location scouted; probe death reduces signal quality
"""

from typing import TYPE_CHECKING

import numpy as np
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from ares.consts import UnitRole

from cython_extensions import cy_dijkstra
from bot.constants import RUSH_SPEED


if TYPE_CHECKING:
    from bot.bot import PiG_Bot


def compute_rush_distance_tier(bot: "PiG_Bot") -> str:
    """
    Compute rush distance tier based on ground path time at zergling speed.
    
    Uses Dijkstra pathfinding from our start location to enemy start location,
    offset by 3 tiles to avoid blocked building positions.
    
    Returns:
        'short' (≤36s), 'medium' (37-45s), or 'long' (≥46s)
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
    dijkstra_result = cy_dijkstra(cost_grid, targets, checks_enabled=False)
    
    # Get ground distance to enemy position
    ground_distance = dijkstra_result.distance[enemy_x, enemy_y]
    
    if ground_distance <= 0 or ground_distance == float('inf'):
        # Pathfind failed, default to medium tier
        bot._rush_time_seconds = 0.0  # Mark as unknown
        return "medium"
    
    # Use rush speed from constants
    rush_time = ground_distance / RUSH_SPEED
    # Classify into distance tiers
    if rush_time <= 36:
        tier = "short"
    elif rush_time <= 45:
        tier = "medium"
    else:
        tier = "long"
    
    bot._rush_time_seconds = rush_time
    return tier


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
    
    Returns:
        Dictionary with keys:
            - tier: str ('short', 'medium', 'long')
            - natural_absent: bool
            - lings_near_base: int
            - total_lings: int
            - speed_seen: bool
            - scout_died_early: bool
            - saw_natural: bool
    """
    tier = bot.rush_distance_tier
    scout_status = _probe_scout_status(bot)
    
    # Natural timing check (distance-adjusted)
    natural_thresholds = {
        'short': 75,   # 1:15
        'medium': 90,  # 1:30
        'long': 105    # 1:45
    }
    natural_absent = (
        bot.time < natural_thresholds[tier] 
        and not any(bot.enemy_structures.closer_than(15, bot.mediator.get_enemy_nat))
    )
    
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
    Main boolean heuristic for ling rush detection (sticky once triggered).
    
    Uses distance-tiered thresholds and multiple signals:
    - Natural expansion timing
    - Zergling counts (total and near our base)
    - Early speed detection
    - Probe scout death handling
    
    Returns:
        True if ling rush detected, False otherwise (sticky once True)
    """
    # Initialize sticky flag and reason tracking
    if not hasattr(bot, '_ling_rushed_v2'):
        bot._ling_rushed_v2 = False
        bot._ling_rush_reason = None
    
    # Return early if already triggered (sticky)
    if bot._ling_rushed_v2:
        return True
    
    signals = get_ling_rush_signals(bot)
    tier = signals['tier']
    
    # Distance-tiered thresholds
    # Format: (time_window, lings_near_threshold, total_lings_threshold)
    thresholds = {
        'short': {
            'early_time': 90,   # 1:30
            'early_near': 2,
            'early_total': 5,
            'mid_time': 150,    # 2:30
            'mid_near': 3,
            'mid_total': 7,
        },
        'medium': {
            'early_time': 105,  # 1:45
            'early_near': 3,
            'early_total': 5,
            'mid_time': 180,    # 3:00
            'mid_near': 3,
            'mid_total': 7,
        },
        'long': {
            'early_time': 120,  # 2:00
            'early_near': 3,
            'early_total': 6,
            'mid_time': 210,    # 3:30
            'mid_near': 3,
            'mid_total': 8,
        },
    }
    
    t = thresholds[tier]
    
    # Apply scout death penalty (reduce thresholds slightly if scout died without seeing natural)
    scout_penalty = 0.85 if (signals['scout_died_early'] and not signals['saw_natural']) else 1.0
    
    triggered = False
    reason = None
    
    # Rule 1: Early lings near base (pool-first indicators)
    if bot.time < t['early_time']:
        adjusted_near = int(t['early_near'] * scout_penalty)
        adjusted_total = int(t['early_total'] * scout_penalty)
        
        if signals['lings_near_base'] >= adjusted_near:
            triggered = True
            reason = f"12p-prox-early-{tier}"
        elif signals['total_lings'] >= adjusted_total:
            triggered = True
            reason = f"12p-total-early-{tier}"
    
    # Rule 2: Mid-game ling counts (general rush)
    if not triggered and bot.time < t['mid_time']:
        if signals['lings_near_base'] >= t['mid_near']:
            triggered = True
            reason = f"ling-prox-mid-{tier}"
        elif signals['total_lings'] >= t['mid_total']:
            triggered = True
            reason = f"ling-total-mid-{tier}"
    
    # Rule 3: No natural + early lings (strong pool-first signal)
    if not triggered and signals['natural_absent'] and bot.time < t['early_time']:
        if signals['total_lings'] >= max(4, int(t['early_total'] * 0.8)):
            triggered = True
            reason = f"no-nat-{tier}"
    
    # Rule 4: Early speed detection (13/12 or 14/14 speedling)
    if not triggered and signals['speed_seen'] and bot.time < 180.0:
        if signals['total_lings'] >= 6:
            triggered = True
            reason = f"early-speed-{tier}"
    
    # Rule 5: Very early single ling (proxy hatch or extreme cheese)
    if not triggered and bot.time < 105.0 and signals['lings_near_base'] >= 1:
        triggered = True
        reason = f"very-early-ling-{tier}"
    
    if triggered:
        bot._ling_rushed_v2 = True
        bot._ling_rush_reason = reason
        print(f"{bot.time_formatted}: Ling rush detected! Reason: {reason}")
    
    return bot._ling_rushed_v2
