"""Blink Stalker chase logic — finish retreating high-value enemies.

Purpose: Detect enemy retreat via cy_is_facing, commit a small blink group to
    chase down the highest-value retreating target, and execute the pursuit
    using StutterGroupForward with aggressive blink gap-closing.
Key Decisions: Reuses damage math and candidate selection from group_snipe.
    Skip-set pattern (bot._chase_committed) same as snipe. cy_is_facing for
    retreat detection — simple, instant, no frame-to-frame state.
Limitations: cy_is_facing can false-positive on units turning briefly. The
    >=50% threshold mitigates this. No position-delta tracking (add later if needed).
"""

from typing import Optional, Union

import numpy as np

from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import (
    AMoveGroup,
    GroupUseAbility,
    StutterGroupForward,
)

from cython_extensions import (
    cy_center,
    cy_distance_to,
    cy_is_facing,
    cy_range_vs_target,
)

from bot.combat.group_snipe import effective_value
from bot.constants import (
    CHASE_MIN_VALUE,
    CHASE_TIMEOUT_FRAMES,
    CHASE_BLINK_GAP_THRESHOLD,
)

# Fraction of enemies that must face away from squad to count as "retreating"
CHASE_RETREAT_FACING_RATIO = 0.5


def is_enemy_retreating(
    enemies: Union[Units, list[Unit]],
    squad_center: Point2,
) -> bool:
    """Check if the enemy group is retreating from our squad.

    Uses cy_is_facing: an enemy NOT facing us (~40° cone toward us) counts
    as "retreating". If >=50% of enemies face away, group is retreating.

    Perf: O(n) where n = enemies. One cy_is_facing call per unit (~200ns each).
    """
    if not enemies:
        return False

    facing_away = sum(
        1 for e in enemies
        if not cy_is_facing(e, squad_center, angle_error=0.7)
    )
    return facing_away >= len(enemies) * CHASE_RETREAT_FACING_RATIO


def find_chase_target(
    enemies: Union[Units, list[Unit]],
    squad_center: Point2,
    min_value: float = CHASE_MIN_VALUE,
) -> Optional[Unit]:
    """Find the best retreating enemy to chase down.

    Filters to retreating enemies above min_value. Returns the
    highest-value retreating target, or None.

    All available squad stalkers will chase — no minimum-needed check.

    Perf: O(e) value+facing scan.
    """
    best_val = min_value
    best_target: Optional[Unit] = None

    for enemy in enemies:
        val = effective_value(enemy)
        if val < best_val:
            continue
        if cy_is_facing(enemy, squad_center, angle_error=0.7):
            continue  # Facing us — not retreating
        best_val = val
        best_target = enemy

    return best_target


def try_commit_chase(
    bot,
    squad_id: str,
    enemies: Union[Units, list[Unit]],
    squad_stalkers: list[Unit],
    squad_position: Point2,
    can_engage: bool,
) -> bool:
    """Evaluate and commit a chase for this squad.

    Called per squad per frame from control_main_army, BEFORE per-unit micro.
    Only commits if we're winning (can_engage) and enemies are retreating.

    Returns True if a chase was committed this frame.

    Perf: O(e + s) where e = enemies, s = stalkers.
    """
    if not can_engage:
        return False

    # Don't stack — only one active chase per squad
    if squad_id in bot._chase_state:
        return False

    if not squad_stalkers:
        return False

    # Check if enemy group is retreating
    if not is_enemy_retreating(enemies, squad_position):
        return False

    target = find_chase_target(
        enemies=enemies,
        squad_center=squad_position,
    )
    if target is None:
        return False

    # All available stalkers join the chase
    candidate_tags = {s.tag for s in squad_stalkers}
    game_loop = bot.state.game_loop

    # Commit: register in skip-set with TTL
    expiry = game_loop + CHASE_TIMEOUT_FRAMES
    for tag in candidate_tags:
        bot._chase_committed[tag] = expiry

    bot._chase_state[squad_id] = {
        "target_tag": target.tag,
        "stalker_tags": candidate_tags,
        "retreat_point": squad_position,
        "commit_frame": game_loop,
    }

    return True


def execute_chase(
    bot,
    squad_id: str,
    grid: np.ndarray,
    nearby_enemies: Union[Units, list[Unit]],
) -> None:
    """Execute chase for an active pursuit.

    nearby_enemies is the same all_close used by the combat loop — scoped
    to UNIT_ENEMY_DETECTION_RANGE. When the target leaves that range,
    it won't be found here and the chase cleans up naturally.

    Each frame:
      target not in nearby  → cleanup (left detection range)
      sim flips to losing   → cleanup
      target far + blink ok → blink to close gap
      else                  → StutterGroupForward (chase and shoot)
    """
    if squad_id not in bot._chase_state:
        return

    info = bot._chase_state[squad_id]
    target_tag = info["target_tag"]
    stalker_tags: set[int] = info["stalker_tags"]
    game_loop = bot.state.game_loop

    # Resolve live stalkers
    stalkers = [u for u in bot.units if u.tag in stalker_tags]
    if not stalkers:
        _cleanup_chase(bot, squad_id)
        return

    tags = {s.tag for s in stalkers}

    # Resolve target from nearby enemies only (not global)
    target: Optional[Unit] = None
    for u in nearby_enemies:
        if u.tag == target_tag:
            target = u
            break

    # Target left detection range, died, or went invisible → done
    if target is None or not target.is_visible:
        _cleanup_chase(bot, squad_id)
        return

    stalker_center = Point2(cy_center(stalkers))
    dist_to_target = cy_distance_to(stalker_center, target.position)

    # Combat sim check — abort if we're now losing
    enemy_list = list(nearby_enemies) if nearby_enemies else []
    if enemy_list:
        from ares.consts import EngagementResult
        sim_result = bot.mediator.can_win_fight(
            own_units=Units(stalkers, bot),
            enemy_units=Units(enemy_list, bot),
            workers_do_no_damage=True,
        )
        losing = {
            EngagementResult.LOSS_DECISIVE,
            EngagementResult.LOSS_OVERWHELMING,
            EngagementResult.LOSS_EMPHATIC,
        }
        if sim_result in losing:
            _cleanup_chase(bot, squad_id)
            return

    maneuver = CombatManeuver()

    # Blink gap-close: if target is pulling away beyond range + threshold
    stalker_range = cy_range_vs_target(stalkers[0], target)
    if dist_to_target > stalker_range + CHASE_BLINK_GAP_THRESHOLD:
        # Check if any stalker has blink ready
        has_blink = any(
            AbilityId.EFFECT_BLINK_STALKER in s.abilities for s in stalkers
        )
        if has_blink:
            blink_stalkers = [
                s for s in stalkers
                if AbilityId.EFFECT_BLINK_STALKER in s.abilities
            ]
            blink_tags = {s.tag for s in blink_stalkers}
            maneuver.add(GroupUseAbility(
                ability=AbilityId.EFFECT_BLINK_STALKER,
                group=blink_stalkers,
                group_tags=blink_tags,
                target=target.position,
                sync_command=True,
            ))
            bot.register_behavior(maneuver)
            return

    # Normal chase: stutter forward toward target
    maneuver.add(StutterGroupForward(
        group=stalkers,
        group_tags=tags,
        group_position=stalker_center,
        target=target.position,
        enemies=Units(nearby_enemies, bot) if nearby_enemies else Units([], bot),
    ))
    bot.register_behavior(maneuver)


def _cleanup_chase(bot, squad_id: str) -> None:
    """Remove all chase state for a squad, releasing stalkers back to normal micro."""
    if squad_id not in bot._chase_state:
        return
    state_info = bot._chase_state[squad_id]
    for tag in state_info["stalker_tags"]:
        bot._chase_committed.pop(tag, None)
    del bot._chase_state[squad_id]
