"""Combat management package for StarCraft II bot.

Handles all combat-related functionality including unit control, targeting, and engagement logic.
"""

from bot.constants import (
    ATTACKING_SQUAD_RADIUS,
    DEFENDER_SQUAD_RADIUS,
    COMMON_UNIT_IGNORE_TYPES,
    DISRUPTOR_IGNORE_TYPES,
    PRIORITY_TARGET_TYPES,
    MELEE_RANGE_THRESHOLD,
    STAY_AGGRESSIVE_DURATION,
    TARGET_LOCK_DISTANCE,
)

from bot.combat.combat import (
    control_main_army,
    gatekeeper_control,
    warp_prism_follower,
    handle_attack_toggles,
    attack_target,
    fallback_target,
    control_base_defenders,
    manage_defensive_unit_roles,
)

from bot.combat.unit_micro import (
    micro_ranged_unit,
    micro_melee_unit,
    micro_zealot,
    micro_disruptor,
    micro_defender_unit,
    get_priority_targets,
)

__all__ = [
    # Constants
    "ATTACKING_SQUAD_RADIUS",
    "DEFENDER_SQUAD_RADIUS",
    "COMMON_UNIT_IGNORE_TYPES",
    "DISRUPTOR_IGNORE_TYPES",
    "PRIORITY_TARGET_TYPES",
    "MELEE_RANGE_THRESHOLD",
    "STAY_AGGRESSIVE_DURATION",
    "TARGET_LOCK_DISTANCE",
    # Combat control functions
    "control_main_army",
    "gatekeeper_control",
    "warp_prism_follower",
    "handle_attack_toggles",
    "attack_target",
    "fallback_target",
    "control_base_defenders",
    "manage_defensive_unit_roles",
    # Unit micro functions
    "micro_ranged_unit",
    "micro_melee_unit",
    "micro_zealot",
    "micro_disruptor",
    "micro_defender_unit",
    "get_priority_targets",
]
