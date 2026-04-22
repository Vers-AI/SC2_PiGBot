"""Group blink-snipe helper functions and Snipe-A coordinator for Stalker micro.

Purpose: Damage math, target evaluation, stalker selection, and coordinated walk-in/blink-out
    sniping. Distance-based logic (approach → fire → blink retreat), no frame counting.
Key Decisions: Reuses existing TACTICAL_BONUS + army_value for target value; uses python-sc2's
    calculate_damage_vs_target for exact damage math; cy_* for all hot-path geometry.
    Skip-set pattern (bot._snipe_committed) prevents per-unit micro from overriding group commands.
Limitations: calculate_damage_vs_target doesn't model Guardian Shield on enemies we haven't
    attacked yet (BuffId check). Overkill buffer compensates.
"""

from math import ceil
from typing import Optional, Union

import numpy as np

from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import AMoveGroup, GroupUseAbility, PathGroupToTarget
from ares.dicts.unit_data import UNIT_DATA

from cython_extensions import (
    cy_center,
    cy_closer_than,
    cy_distance_to,
    cy_in_attack_range,
    cy_range_vs_target,
    cy_sorted_by_distance_to,
)
from cython_extensions.numpy_helper import cy_all_points_below_max_value

from bot.combat.target_scoring import TACTICAL_BONUS, TYPE_VALUE_SCALE, DEFAULT_TYPE_VALUE
from ares.consts import EngagementResult

from bot.constants import (
    SNIPE_MIN_TARGET_VALUE,
    SNIPE_OVERKILL_BUFFER_MOBILE,
    SNIPE_OVERKILL_BUFFER_STATIC,
    SNIPE_EXIT_FRAMES,
    SNIPE_COMMIT_COOLDOWN,
    SNIPE_APPROACH_RANGE_BUFFER,
)

# Combat sim results that allow snipe commitment
SNIPE_SIM_THRESHOLD = {EngagementResult.TIE, EngagementResult.VICTORY_MARGINAL,
                       EngagementResult.VICTORY_CLOSE, EngagementResult.VICTORY_DECISIVE,
                       EngagementResult.VICTORY_OVERWHELMING, EngagementResult.VICTORY_EMPHATIC}

# Max grid value along the approach corridor (before firing range).
# Safe cell = 1.0; enemy influence adds cost. ~15 allows walking past
# light threats but blocks marching through a full army.
SNIPE_CORRIDOR_MAX_VALUE = 15.0
SNIPE_CORRIDOR_SAMPLES = 4  # number of evenly-spaced sample points




def effective_value(target: Unit) -> float:
    """Compute the effective tactical value of an enemy unit.

    Combines ARES army_value (economic) with TACTICAL_BONUS (gameplay impact).
    Same formula used in score_target() for consistency.

    Perf: ~200ns (dict lookups only, no iteration).
    """
    data = UNIT_DATA.get(target.type_id)
    base = (data["army_value"] * TYPE_VALUE_SCALE) if data else DEFAULT_TYPE_VALUE
    return base + TACTICAL_BONUS.get(target.type_id, 0.0)


def damage_per_volley(stalker: Unit, target: Unit) -> float:
    """Compute the damage a single stalker deals to target in one attack cycle.

    Uses python-sc2's calculate_damage_vs_target which accounts for:
    - Upgrades (attack + armor)
    - Bonus damage vs armor types (e.g., Stalker +bonus vs Armored)
    - Guardian Shield buff on target
    - Multi-attack weapons
    - Air/ground targeting

    Returns 0.0 if stalker can't attack the target.

    Perf: ~2µs (python-sc2 property access + weapon iteration).
    """
    dmg, _speed, _range = stalker.calculate_damage_vs_target(target)
    return float(dmg)


def stalkers_needed_to_kill(
    stalker: Unit,
    target: Unit,
) -> int:
    """Calculate the minimum number of stalkers needed to kill target in one volley.

    Uses a representative stalker for damage calculation (all stalkers in a squad
    should have the same upgrades). Adds an overkill buffer: +1 for mobile targets
    (they might dodge), +0 for static targets (sieged tanks, burrowed units).

    Returns 0 if stalker can't attack the target (e.g., can't hit air).

    Perf: ~2µs (one calculate_damage_vs_target call + arithmetic).
    """
    dmg = damage_per_volley(stalker, target)
    if dmg <= 0:
        return 0

    target_hp = target.health + target.shield
    raw_needed = ceil(target_hp / dmg)

    # Static targets (sieged, burrowed, phased) can't dodge — no overkill buffer needed
    buffer = (
        SNIPE_OVERKILL_BUFFER_STATIC
        if target.movement_speed == 0
        else SNIPE_OVERKILL_BUFFER_MOBILE
    )
    return raw_needed + buffer


def find_snipe_candidates(
    stalkers: list[Unit],
    target: Unit,
    needed: int,
    max_range: float = 20.0,
) -> list[Unit]:
    """Select the minimum stalkers needed for a snipe, from those eligible.

    Filters:
    - Blink ready (EFFECT_BLINK_STALKER in abilities)
    - Within max_range of the target

    Weapons are not checked here — stalkers use pure move during approach,
    so any prior cooldowns will expire before they reach firing range.

    Sorts by: distance to target (closest first) to minimize travel time.
    Returns up to `needed` stalkers. Returns empty list if not enough qualify.

    Perf: O(n) filter + O(n log n) sort where n = stalkers in range. Typically n < 20.
    """
    nearby: list[Unit] = cy_closer_than(stalkers, max_range, target.position)

    eligible: list[Unit] = [
        s for s in nearby
        if AbilityId.EFFECT_BLINK_STALKER in s.abilities
    ]

    if len(eligible) < needed:
        return []

    sorted_by_dist: list[Unit] = cy_sorted_by_distance_to(eligible, target.position)
    return sorted_by_dist[:needed]


def is_snipe_a_eligible(stalker_sample: Unit, target: Unit) -> bool:
    """Check if Snipe-A (walk-in, blink-out) is viable for this target.

    Snipe-A requires we can walk into weapon range without being out-ranged.
    True when target's weapon range ≤ stalker range + buffer.

    Perf: ~300ns (two cy_range_vs_target calls).
    """
    our_range = cy_range_vs_target(stalker_sample, target)
    their_range = cy_range_vs_target(target, stalker_sample)
    return their_range <= our_range + SNIPE_APPROACH_RANGE_BUFFER


def _is_corridor_safe(
    bot,
    squad_center: Union[Point2, tuple[float, float]],
    target_pos: Point2,
    stalker_sample: Unit,
    target: Unit,
) -> bool:
    """Check if the approach corridor is safe enough for a snipe walk-in.

    Samples points along the path from squad_center to the edge of firing
    range around the target. The last ~7 tiles near the target (where the
    stalkers will actually shoot) are excluded — danger there is expected.

    If the stalkers are already within firing range, the corridor is empty
    and this always returns True.

    Perf: O(SNIPE_CORRIDOR_SAMPLES) grid lookups (~4 us total).
    """
    firing_range = cy_range_vs_target(stalker_sample, target) + 1.0
    sc = Point2(squad_center) if not isinstance(squad_center, Point2) else squad_center
    total_dist = cy_distance_to(sc, target_pos)

    # Already close enough — no corridor to check
    corridor_length = total_dist - firing_range
    if corridor_length <= 2.0:
        return True

    # Sample evenly-spaced points along the corridor (before firing range)
    grid: np.ndarray = bot.mediator.get_ground_grid
    dx = target_pos.x - sc.x
    dy = target_pos.y - sc.y
    # Normalize direction
    inv_dist = 1.0 / total_dist
    dx *= inv_dist
    dy *= inv_dist

    sample_points: list[tuple[int, int]] = []
    for i in range(1, SNIPE_CORRIDOR_SAMPLES + 1):
        t = (corridor_length * i / (SNIPE_CORRIDOR_SAMPLES + 1))
        px = int(sc.x + dx * t)
        py = int(sc.y + dy * t)
        sample_points.append((px, py))

    return cy_all_points_below_max_value(
        grid, SNIPE_CORRIDOR_MAX_VALUE, sample_points
    )


def find_best_snipe_target(
    bot,
    enemies: Union[Units, list[Unit]],
    stalker_sample: Unit,
    squad_stalkers: list[Unit],
    min_value: float = SNIPE_MIN_TARGET_VALUE,
) -> Optional[tuple[Unit, int, list[Unit]]]:
    """Find the best target for a group blink snipe.

    Evaluates all enemies by effective_value, filters to Snipe-A eligible targets,
    checks damage math + candidate count, then runs combat sim to verify the snipe
    squad can survive against nearby enemies (TIE or better).

    Returns (target, needed_count, selected_stalkers) or None if no viable snipe.

    This is the main entry point called per-squad per-frame.

    Perf: O(e) value scan + O(1) damage calc + O(s) candidate selection + 1 combat sim.
    where e = enemies, s = squad stalkers. Typically e < 30, s < 15.
    """
    valued: list[tuple[float, Unit]] = []
    for enemy in enemies:
        val = effective_value(enemy)
        if val < min_value:
            continue
        # Snipe-A only: target must be walk-in eligible
        if not is_snipe_a_eligible(stalker_sample, enemy):
            continue
        valued.append((val, enemy))

    if not valued:
        return None

    # Sort by value descending, then by current HP ascending (easiest kill first)
    valued.sort(key=lambda x: (x[0], -(x[1].health + x[1].shield)), reverse=True)

    for _val, target in valued:
        needed = stalkers_needed_to_kill(stalker_sample, target)
        if needed <= 0:
            continue

        candidates = find_snipe_candidates(
            squad_stalkers, target, needed,
        )
        if not candidates:
            continue

        # Combat sim gate: can the snipe squad survive against nearby enemies?
        sim_result = bot.mediator.can_win_fight(
            own_units=Units(candidates, bot),
            enemy_units=Units(list(enemies), bot),
            workers_do_no_damage=True,
        )
        if sim_result not in SNIPE_SIM_THRESHOLD:
            continue

        # Approach corridor safety: don't walk through an army to reach target.
        # Only checks the path BEFORE firing range — the last ~7 tiles near
        # the target are expected to be hot and are OK.
        squad_center = cy_center([c.position for c in candidates])
        if not _is_corridor_safe(bot, squad_center, target.position, stalker_sample, target):
            continue

        return target, needed, candidates

    return None


# ===== SNIPE-A STATE MACHINE =====

def try_commit_snipe(
    bot,
    squad_id: str,
    enemies: Union[Units, list[Unit]],
    squad_stalkers: list[Unit],
    squad_position: Point2,
) -> bool:
    """Evaluate and commit a Snipe-A (walk-in, blink-out) for this squad.

    Called once per squad per frame from control_main_army, BEFORE per-unit micro.
    If a snipe is committed, adds tags to bot._snipe_committed and creates
    a _snipe_state entry. Returns True if a snipe was committed this frame.

    Gate checks applied here:
    - Cooldown: no snipe if squad committed within SNIPE_COMMIT_COOLDOWN frames
    - No existing snipe in progress for this squad
    - Target found via find_best_snipe_target (value + range + damage math + combat sim)

    Perf: O(e + s) where e = enemies, s = stalkers. Typically < 50 units total.
    """
    game_loop = bot.state.game_loop

    # Cooldown check: don't re-commit too soon after a previous snipe
    last_commit = bot._snipe_squad_cooldown.get(squad_id, 0)
    if game_loop - last_commit < SNIPE_COMMIT_COOLDOWN:
        return False

    # Don't stack snipes — only one active snipe per squad
    if squad_id in bot._snipe_state:
        return False

    # Need at least one stalker with blink to even evaluate
    if not squad_stalkers:
        return False

    stalker_sample = squad_stalkers[0]

    result = find_best_snipe_target(
        bot=bot,
        enemies=enemies,
        stalker_sample=stalker_sample,
        squad_stalkers=squad_stalkers,
    )
    if result is None:
        return False

    target, needed, candidates = result
    candidate_tags = {s.tag for s in candidates}

    # Raw kill count (without overkill buffer) — minimum stalkers to one-shot
    dmg = damage_per_volley(stalker_sample, target)
    raw_needed = ceil((target.health + target.shield) / dmg) if dmg > 0 else needed

    # Commit: register in skip-set with TTL
    expiry = game_loop + SNIPE_EXIT_FRAMES + 200  # generous TTL for full lifecycle
    for tag in candidate_tags:
        bot._snipe_committed[tag] = expiry

    # Record snipe state for this squad
    bot._snipe_state[squad_id] = {
        "target_tag": target.tag,
        "stalker_tags": candidate_tags,
        "retreat_point": squad_position,
        "commit_frame": game_loop,
        "min_to_kill": raw_needed,
    }
    bot._snipe_squad_cooldown[squad_id] = game_loop

    return True


def execute_snipe_a(bot, squad_id: str, grid: np.ndarray) -> None:
    """Execute Snipe-A (walk-in, blink-out) for an active snipe.

    Each frame reads unit state and decides:
      not in range        → PathGroupToTarget (pure move, weapons recharge)
      in range, shooting  → AMoveGroup (volley)
      all weapons cooling → blink retreat (shots are out)
      target dead/lost    → blink retreat (job done or abort)
      stalkers all dead   → cleanup only (nobody to blink)
    """
    if squad_id not in bot._snipe_state:
        return

    info = bot._snipe_state[squad_id]
    target_tag = info["target_tag"]
    stalker_tags: set[int] = info["stalker_tags"]
    retreat = info["retreat_point"]
    game_loop = bot.state.game_loop

    # Resolve live stalkers
    stalkers = [u for u in bot.units if u.tag in stalker_tags]
    if not stalkers:
        _cleanup_snipe(bot, squad_id)
        return

    tags = {s.tag for s in stalkers}

    # Resolve target — if dead/lost, blink retreat immediately
    target: Optional[Unit] = None
    for u in bot.all_enemy_units:
        if u.tag == target_tag:
            target = u
            break

    if target is None or not target.is_visible:
        maneuver = CombatManeuver()
        maneuver.add(GroupUseAbility(
            ability=AbilityId.EFFECT_BLINK_STALKER,
            group=stalkers,
            group_tags=tags,
            target=retreat,
            sync_command=True,
        ))
        bot.register_behavior(maneuver)
        _cleanup_snipe(bot, squad_id)
        return

    in_range = [s for s in stalkers if cy_in_attack_range(s, [target], bonus_distance=0.5)]
    # "Enough" = at least the raw number needed to one-shot the target
    enough_in_range = len(in_range) >= info["min_to_kill"]

    maneuver = CombatManeuver()

    # --- Approach: pure move (no attacking) so weapons recharge en route ---
    if not enough_in_range:
        stalker_center = Point2(cy_center(stalkers))
        maneuver.add(PathGroupToTarget(
            start=stalker_center,
            group=stalkers,
            group_tags=tags,
            grid=grid,
            target=target.position,
            success_at_distance=2.0,
        ))
        bot.register_behavior(maneuver)
        return

    # All weapons on cooldown = all shots have left the barrel → blink
    # Only check in-range stalkers; stragglers blink regardless
    if all(not s.weapon_ready for s in in_range):
        maneuver.add(GroupUseAbility(
            ability=AbilityId.EFFECT_BLINK_STALKER,
            group=stalkers,
            group_tags=tags,
            target=retreat,
            sync_command=True,
        ))
        bot.register_behavior(maneuver)
        _cleanup_snipe(bot, squad_id)
        return

    # Weapons still ready / firing — keep attacking
    maneuver.add(AMoveGroup(group=in_range, group_tags={s.tag for s in in_range}, target=target))
    bot.register_behavior(maneuver)


def _cleanup_snipe(bot, squad_id: str) -> None:
    """Remove all snipe state for a squad, releasing stalkers back to normal micro."""
    if squad_id not in bot._snipe_state:
        return
    state_info = bot._snipe_state[squad_id]
    for tag in state_info["stalker_tags"]:
        bot._snipe_committed.pop(tag, None)
    del bot._snipe_state[squad_id]
