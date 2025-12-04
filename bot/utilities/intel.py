"""
Utility functions for gathering intelligence about the enemy.

Purpose: Track enemy intel quality and cannon rush detection
Key Decisions: Use unit.age and unit.is_memory for staleness tracking
Limitations: Ghost units expire after 30s in ARES UnitMemoryManager
"""
from typing import TYPE_CHECKING

import numpy as np
from map_analyzer import MapData
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2

from ares.consts import WORKER_TYPES

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


# Choke detection config
RAMP_CHOKE_RADIUS = 6.0   # Radius around ramp top/bottom
MAP_CHOKE_RADIUS = 10.0   # Radius around map_analyzer chokes
CHOKE_GRID_WEIGHT = 10.0  # Weight to mark choke zones (> 1.0 = choke area)


def create_choke_grid(bot: "PiG_Bot") -> np.ndarray:
    """
    Create a grid marking choke/ramp areas for O(1) formation skip detection.
    
    Call this once in on_start and store as bot.choke_grid.
    Cells with value > 1.0 indicate "near choke, skip formation logic".
    
    Args:
        bot: Bot instance with access to map_data and game_info
        
    Returns:
        np.ndarray: Grid where values > 1.0 mark choke/ramp zones
    """
    map_data: MapData = bot.mediator.get_map_data_object
    grid = map_data.get_pyastar_grid()
    
    # Mark ramps (top and bottom centers)
    for ramp in bot.game_info.map_ramps:
        grid = map_data.add_cost(
            ramp.top_center, RAMP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )
        grid = map_data.add_cost(
            ramp.bottom_center, RAMP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )
    
    # Mark map_analyzer choke points
    for choke in map_data.map_chokes:
        grid = map_data.add_cost(
            choke.center, MAP_CHOKE_RADIUS, grid, CHOKE_GRID_WEIGHT
        )
    
    return grid


def is_near_choke(choke_grid: np.ndarray, position: Point2) -> bool:
    """
    O(1) check if position is near a choke/ramp using precomputed grid.
    
    Args:
        choke_grid: Precomputed choke grid from create_choke_grid()
        position: Position to check (typically army_center)
        
    Returns:
        True if near choke (skip formation), False otherwise
    """
    pos = position.rounded
    return choke_grid[pos[0], pos[1]] > 1.0


def get_enemy_cannon_rushed(bot, detection_radius: float = 25.0) -> bool:
    """Check if the enemy is cannon rushing.
    
    Args:
        bot: The bot instance
        detection_radius: Distance in game units to check around each base (default: 25.0)
        
    Returns:
        bool: True if cannon rush is detected, False otherwise
    """
    # Only check against Protoss
    if bot.enemy_race != Race.Protoss:
        return False
        
    # Only check in early game (before 3 minutes)
    if bot.time > 180.0:  # 3 minutes
        return False
    
    try:
        # Get main base and natural expansion positions
        main_base = bot.start_location
        natural_expansion = bot.mediator.get_own_nat  # This gets the natural expansion position
        
        # Look for enemy pylons and cannons near our bases
        enemy_structures = bot.enemy_structures.filter(
            lambda s: s.type_id in {UnitID.PYLON, UnitID.PHOTONCANNON} and 
                     (s.distance_to(main_base) < detection_radius or 
                      s.distance_to(natural_expansion) < detection_radius)
        )
        
        # Count pylons and cannons
        pylon_count = sum(1 for s in enemy_structures if s.type_id == UnitID.PYLON)
        cannon_count = sum(1 for s in enemy_structures if s.type_id == UnitID.PHOTONCANNON)
        
        # If we see cannons or multiple pylons near our bases, it's likely a cannon rush
        if cannon_count > 0 or pylon_count > 1:
            return True
            
        return False
        
    except Exception as e:
        print(f"Error in cannon rush detection: {e}")
        return False


def get_enemy_intel_quality(bot: "PiG_Bot") -> dict:
    """
    Analyze the freshness of our enemy army intel.
    
    Uses unit.age and unit.is_memory from python-sc2 to determine
    how trustworthy our combat simulation data is.
    
    Returns:
        dict with keys:
            - has_intel: bool - have we ever seen enemy army?
            - avg_age: float - average age of enemy unit data (seconds)
            - visible_count: int - units we can currently see
            - memory_count: int - units from memory (not visible)
            - freshness: float - 0-1 score (1 = all fresh, 0 = all stale)
    """
    # Get enemy army excluding workers from CACHED enemy (includes memory units from ARES)
    cached_enemy = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES
    ]
    
    # Also check currently visible enemy units (raw observation, no memory)
    # bot.enemy_units comes directly from game observation
    current_visible = [
        u for u in bot.enemy_units
        if u.type_id not in WORKER_TYPES and not u.is_structure
    ]
    
    # If we have neither cached nor current visible enemies
    if not cached_enemy and not current_visible:
        # If we've EVER seen enemy army before, 0 enemies is GOOD intel (we know they have nothing)
        # Only return "no intel" if we've never seen enemy army at allP
        if bot._enemy_army_ever_seen:
            return {
                "has_intel": True,  # We HAVE intel: enemy has no visible army!
                "avg_age": 0.0,     # Current observation
                "visible_count": 0,
                "memory_count": 0,
                "freshness": 1.0,   # 0 enemies visible = perfectly fresh intel
            }
        else:
            return {
                "has_intel": False,  # Never seen enemy army, truly blind
                "avg_age": float('inf'),
                "visible_count": 0,
                "memory_count": 0,
                "freshness": 0.0,
            }
    
    # Count visible as units we've seen recently (age < 3 seconds)
    # 3s is forgiving for active combat where snapshots update slightly delayed
    # This is more reliable than is_memory which can be inconsistent
    VISIBLE_AGE_THRESHOLD = 3.0
    visible = [u for u in cached_enemy if u.age < VISIBLE_AGE_THRESHOLD]
    memory = [u for u in cached_enemy if u.age >= VISIBLE_AGE_THRESHOLD]
    
    # Fallback: if cached shows no visible but we have current_visible, use that count
    if not visible and current_visible:
        visible = current_visible
    
    total_count = len(visible) + len(memory)
    if total_count == 0:
        return {
            "has_intel": False,
            "avg_age": float('inf'),
            "visible_count": 0,
            "memory_count": 0,
            "freshness": 0.0,
        }
    
    # Calculate average age of all enemy units we know about
    all_units = cached_enemy if cached_enemy else current_visible
    avg_age = sum(u.age for u in all_units) / len(all_units) if all_units else 0.0
    
    # Freshness score: 1.0 = all visible now, decays over 30 seconds
    # (30s matches ARES UnitMemoryManager expiration)
    STALENESS_WINDOW = 30.0
    freshness = sum(max(0, 1 - u.age / STALENESS_WINDOW) for u in all_units) / len(all_units)
    
    return {
        "has_intel": True,
        "avg_age": avg_age,
        "visible_count": len(visible),
        "memory_count": len(memory),
        "freshness": freshness,
    }


def update_enemy_intel_tracking(bot: "PiG_Bot") -> None:
    """
    Update enemy intel tracking flags and intel urgency. Call this every frame.
    
    Sets:
        - bot._enemy_army_ever_seen: sticky flag, True once we've seen enemy army
        - bot._last_enemy_army_visible_time: last time we had direct vision of enemy army
        - bot._intel_urgency: 0-1 score, builds when stale, decays when fresh
    """
    # Get enemy army excluding workers
    enemy_army = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES
    ]
    
    # Check if any enemy army is currently visible (not memory)
    visible_army = [u for u in enemy_army if not u.is_memory]
    
    if visible_army:
        # We currently see enemy army
        bot._enemy_army_ever_seen = True
        bot._last_enemy_army_visible_time = bot.time
    elif enemy_army:
        # We have memory of enemy army but can't see them now
        bot._enemy_army_ever_seen = True
    
    # Update intel urgency (gradual build/decay to avoid oscillation)
    # Only track urgency after we've seen enemy army at least once
    if bot._enemy_army_ever_seen:
        intel = get_enemy_intel_quality(bot)
        freshness = intel["freshness"]
        
        # Stale threshold matches combat.py (< 0.3 = STALE/BLIND)
        STALE_THRESHOLD = 0.3
        URGENCY_BUILD_RATE = 0.02   # How fast urgency builds (per frame, ~0.4/sec at 22fps)
        URGENCY_DECAY_RATE = 0.005  # How fast urgency decays (slower than build)
        
        if freshness < STALE_THRESHOLD:
            # Intel is stale - build urgency
            bot._intel_urgency = min(1.0, bot._intel_urgency + URGENCY_BUILD_RATE)
        else:
            # Intel is fresh - slowly decay urgency
            bot._intel_urgency = max(0.0, bot._intel_urgency - URGENCY_DECAY_RATE)
