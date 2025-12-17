"""System-wide constants and configuration values.

Purpose: Centralize all magic numbers, unit filters, and tunable parameters across the entire bot
Key Decisions: Immutable constants to prevent accidental modification, organized by subsystem
Limitations: None - pure data definitions

Organization:
- Combat constants (squad radii, detection ranges, timings)
- Unit filtering sets (ignore lists, priority targets)
- Future: Macro constants, scouting parameters, etc.
"""

from sc2.ids.unit_typeid import UnitTypeId

# ===== SQUAD CONFIGURATION =====
ATTACKING_SQUAD_RADIUS = 9.0
"""Squad radius for ATTACKING role units - looser formation for army mobility"""

DEFENDER_SQUAD_RADIUS = 6.0
"""Squad radius for BASE_DEFENDER role units - tighter formation for base defense"""

# ===== UNIT FILTERING =====
COMMON_UNIT_IGNORE_TYPES: set[UnitTypeId] = {
    UnitTypeId.EGG,
    UnitTypeId.LARVA,
    UnitTypeId.CREEPTUMORBURROWED,
    UnitTypeId.CREEPTUMORQUEEN,
    UnitTypeId.CREEPTUMOR,
    UnitTypeId.MULE,
    UnitTypeId.OVERLORD,
    UnitTypeId.OVERSEER,
    UnitTypeId.LOCUSTMP,
    UnitTypeId.LOCUSTMPFLYING,
    UnitTypeId.ADEPTPHASESHIFT,
    UnitTypeId.CHANGELING,
    UnitTypeId.CHANGELINGMARINE,
    UnitTypeId.CHANGELINGZEALOT,
    UnitTypeId.CHANGELINGZERGLING,
}
"""Units to ignore in combat targeting - non-threatening or temporary units"""

DISRUPTOR_IGNORE_TYPES: set[UnitTypeId] = COMMON_UNIT_IGNORE_TYPES | {
    UnitTypeId.SCV,
    UnitTypeId.DRONE,
    UnitTypeId.PROBE,
    UnitTypeId.BROODLING,
}
"""Units to ignore for Disruptor nova targeting - includes workers (too low value)"""

PRIORITY_TARGET_TYPES: set[UnitTypeId] = {
    UnitTypeId.SIEGETANK,
    UnitTypeId.SIEGETANKSIEGED,
    UnitTypeId.COLOSSUS,
}
"""High-priority targets that should be focused first in combat"""

# ===== COMBAT PARAMETERS =====
MELEE_RANGE_THRESHOLD = 3.0
"""Range threshold to classify units as melee vs ranged"""

STAY_AGGRESSIVE_DURATION = 20.0
"""Minimum time (seconds) to stay committed to an attack before reconsidering"""

TARGET_LOCK_DISTANCE = 25.0
"""Distance threshold for switching attack targets - prevents oscillation"""

# ===== COMBAT SIMULATOR THRESHOLDS =====
SIEGE_TANK_SUPPLY_ADVANTAGE_REQUIRED = 2.0
"""Supply advantage multiplier needed to attack into siege tanks (combat sim often underestimates them)"""

SQUAD_NEARBY_FRIENDLY_RANGE_SQ = 255.0
"""Squared distance (16^2) to find nearby friendly units for squad-level combat simulation"""

# ===== DISTANCE THRESHOLDS =====
UNSAFE_GROUND_CHECK_RADIUS = 8.0
"""Radius to search for safe spots when unit is on unsafe ground"""

DISRUPTOR_SQUAD_FOLLOW_DISTANCE = 5.0
"""Maximum distance disruptor should be from squad center"""

DISRUPTOR_SQUAD_TARGET_DISTANCE = 4.0
"""Target distance for disruptor to maintain from squad"""

DISRUPTOR_FORWARD_OFFSET = 8.0
"""Distance Disruptor positions ahead of army toward target"""

STRUCTURE_ATTACK_RANGE = 12.0
"""Range to detect nearby enemy structures"""

PROXIMITY_STICKY_DISTANCE_SQ = 450.0
"""Squared distance (21.2^2) for proximity stickiness to structures"""

MAP_CROSSING_DISTANCE_SQ = 49.0
"""Squared distance (7^2) threshold for safe map crossing with PathUnitToTarget"""

MAP_CROSSING_SUCCESS_DISTANCE = 6.5
"""Success distance for PathUnitToTarget during map crossing"""

# ===== ENGAGEMENT RANGES =====
UNIT_ENEMY_DETECTION_RANGE = 12.0
"""Range to detect enemies around each unit's position (per-unit detection)"""

GATEKEEPER_DETECTION_RANGE = 6.0
"""Range to detect enemies around gatekeeping position"""

GATEKEEPER_MOVE_DISTANCE = 3.0
"""Distance gatekeeper moves when no enemies present"""

# ===== WARP PRISM PARAMETERS =====
WARP_PRISM_FOLLOW_DISTANCE = 15.0
"""Distance threshold for warp prism to morph to phasing mode"""

WARP_PRISM_FOLLOW_OFFSET = 3.0
"""Distance offset behind army center for warp prism positioning"""

WARP_PRISM_UNIT_CHECK_RANGE = 6.5
"""Range to check for units warping in near prism"""

WARP_PRISM_DANGER_DISTANCE = 10.0
"""Danger distance parameter for warp prism pathfinding"""

# ===== UNDER ATTACK DETECTION =====
# OBSERVATION: These thresholds determine when _under_attack flag triggers
# Adjust based on gameplay feedback
UNDER_ATTACK_VALUE_THRESHOLD = 15.0
"""Minimum threat value to trigger under_attack (roughly 3 stalkers worth)"""

UNDER_ATTACK_RATIO_THRESHOLD = 0.4
"""Minimum ratio of enemy army near bases to trigger under_attack (40% of known army)"""

UNDER_ATTACK_CLEAR_VALUE = 5.0
"""Threat value below which under_attack clears (hysteresis to prevent flickering)"""

# ===== EARLY GAME DEFENSE =====
EARLY_GAME_TIME_LIMIT = 600.0
"""Time limit (seconds) for early game defensive positioning (10 minutes)"""

EARLY_GAME_SAFE_GROUND_CHECK_BASES = 1
"""Maximum number of bases to still check for safe ground positioning"""

# ===== RUSH DETECTION =====
RUSH_SPEED = 3.94
"""Rush movement speed (used for rush distance calculations)"""

RUSH_DISTANCE_CALIBRATION = 0.833
"""Calibration factor to match official map rush distances (36s official / 43.2s calculated = 0.833)"""
