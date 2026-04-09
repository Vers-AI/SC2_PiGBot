"""Pre-engagement concave formation via three-group fan-out.

Purpose: Spread ranged ground units into a line before bulk engagements for better concave
Key Decisions: Squad-level, ranged-only, 3-group split (left/center/right), one-shot setup
Limitations: Open terrain only; choke/ramp handling is upstream responsibility
"""

import math
from sc2.position import Point2
from sc2.unit import Unit
from sc2.ids.ability_id import AbilityId
from cython_extensions import cy_distance_to

from bot.constants import (
    CONCAVE_TRIGGER_RANGE,
    CONCAVE_MIN_RANGED_UNITS,
    CONCAVE_FAN_WIDTH_PER_UNIT,
    CONCAVE_MAX_FAN_WIDTH,
    CONCAVE_SPREAD_FRAMES,
    CONCAVE_WEAPON_RANGE_ABORT,
    CONCAVE_RESET_RANGE,
)


def should_form_concave(
    ranged_units: list[Unit],
    squad_center: Point2,
    enemy_center: Point2,
) -> bool:
    """Check if this squad should enter concave formation mode.

    Triggers when enough ranged units exist AND enemy is within trigger range
    but not yet in weapon range (still time to spread).
    """
    if len(ranged_units) < CONCAVE_MIN_RANGED_UNITS:
        return False

    dist_to_enemy = cy_distance_to(squad_center, enemy_center)

    # Too close — just fight, no time to spread
    if dist_to_enemy < CONCAVE_WEAPON_RANGE_ABORT:
        return False

    # Within trigger range — time to spread
    return dist_to_enemy < CONCAVE_TRIGGER_RANGE


def compute_fan_targets(
    squad_center: Point2,
    enemy_center: Point2,
    num_ranged: int,
) -> tuple[Point2, Point2, Point2]:
    """Compute 3 target points for left/center/right fan-out groups.

    Targets are near the enemy but offset laterally so units approach
    from a wide line rather than a clump. A-moving toward these targets
    naturally creates the concave on contact.

    Returns:
        (left_target, center_target, right_target)
    """
    dx = enemy_center.x - squad_center.x
    dy = enemy_center.y - squad_center.y
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.01:
        return enemy_center, enemy_center, enemy_center

    # Perpendicular axis (left/right relative to approach direction)
    perp_x = -dy / length
    perp_y = dx / length

    # Fan width scales with unit count, capped
    fan_width = min(num_ranged * CONCAVE_FAN_WIDTH_PER_UNIT, CONCAVE_MAX_FAN_WIDTH)
    half_width = fan_width / 2.0

    center_target = enemy_center
    left_target = Point2((
        enemy_center.x + perp_x * half_width,
        enemy_center.y + perp_y * half_width,
    ))
    right_target = Point2((
        enemy_center.x - perp_x * half_width,
        enemy_center.y - perp_y * half_width,
    ))

    return left_target, center_target, right_target


def split_into_fan_groups(
    ranged_units: list[Unit],
    squad_center: Point2,
    enemy_center: Point2,
) -> tuple[list[Unit], list[Unit], list[Unit]]:
    """Split ranged units into left/center/right by lateral position.

    Sorts units by their offset on the perpendicular axis (relative to
    the approach direction), then divides into thirds. Extra units go
    to the center group.
    """
    dx = enemy_center.x - squad_center.x
    dy = enemy_center.y - squad_center.y
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.01:
        return [], ranged_units, []

    perp_x = -dy / length
    perp_y = dx / length

    # Compute lateral offset (dot product with perpendicular) per unit
    offsets: list[tuple[float, Unit]] = []
    for unit in ranged_units:
        rel_x = unit.position.x - squad_center.x
        rel_y = unit.position.y - squad_center.y
        lateral = rel_x * perp_x + rel_y * perp_y
        offsets.append((lateral, unit))

    # Sort: most negative = right side, most positive = left side
    offsets.sort(key=lambda x: x[0])

    n = len(offsets)
    third = n // 3
    remainder = n % 3

    # Extra units go to center
    right_end = third
    center_end = third + third + remainder

    right_group = [u for _, u in offsets[:right_end]]
    center_group = [u for _, u in offsets[right_end:center_end]]
    left_group = [u for _, u in offsets[center_end:]]

    return left_group, center_group, right_group


def execute_fan_out(
    bot,
    squad_id: str,
    ranged_units: list[Unit],
    squad_center: Point2,
    enemy_center: Point2,
) -> bool:
    """Issue fan-out move commands for ranged units in a squad.

    Tracks formation state per squad. One-shot: once spreading is done
    (timeout or enemy too close), transitions to normal micro permanently
    for this engagement.

    Args:
        bot: Bot instance (needs give_same_action, state.game_loop)
        squad_id: Unique squad identifier for state tracking
        ranged_units: Ground ranged units in this squad
        squad_center: Current center of the squad
        enemy_center: Center of nearby enemy army

    Returns:
        True if formation is actively spreading (caller should skip
        per-unit ranged micro). False if not active or done.
    """
    # Lazy-init state tracking
    if not hasattr(bot, '_squad_formation_state'):
        bot._squad_formation_state = {}

    state = bot._squad_formation_state.get(squad_id)

    # --- State: not started yet ---
    if state is None:
        if not should_form_concave(ranged_units, squad_center, enemy_center):
            return False
        # Begin spreading
        bot._squad_formation_state[squad_id] = {
            "frame_start": bot.state.game_loop,
            "status": "spreading",
        }
        state = bot._squad_formation_state[squad_id]

    # --- State: already done for this engagement ---
    if state["status"] == "done":
        # Reset if enemies moved far away (new engagement later)
        dist_to_enemy = cy_distance_to(squad_center, enemy_center)
        if dist_to_enemy > CONCAVE_RESET_RANGE:
            del bot._squad_formation_state[squad_id]
        return False

    # --- State: actively spreading ---
    dist_to_enemy = cy_distance_to(squad_center, enemy_center)
    frames_elapsed = bot.state.game_loop - state["frame_start"]

    # Abort: enemy in weapon range or spread timeout
    if dist_to_enemy < CONCAVE_WEAPON_RANGE_ABORT or frames_elapsed > CONCAVE_SPREAD_FRAMES:
        state["status"] = "done"
        return False

    # Compute groups and targets
    left_group, center_group, right_group = split_into_fan_groups(
        ranged_units, squad_center, enemy_center
    )
    left_target, center_target, right_target = compute_fan_targets(
        squad_center, enemy_center, len(ranged_units)
    )

    # Issue A-move commands per sub-group
    # ATTACK (not MOVE) so units fight anything they encounter while spreading
    if left_group:
        left_tags = {u.tag for u in left_group}
        bot.give_same_action(AbilityId.ATTACK, left_tags, left_target)

    if center_group:
        center_tags = {u.tag for u in center_group}
        bot.give_same_action(AbilityId.ATTACK, center_tags, center_target)

    if right_group:
        right_tags = {u.tag for u in right_group}
        bot.give_same_action(AbilityId.ATTACK, right_tags, right_target)

    return True  # Still spreading — caller should skip per-unit ranged micro


def clear_formation_state(bot, squad_id: str) -> None:
    """Clear formation state for a squad (e.g., when no enemies nearby)."""
    if hasattr(bot, '_squad_formation_state') and squad_id in bot._squad_formation_state:
        del bot._squad_formation_state[squad_id]
