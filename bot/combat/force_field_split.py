"""Force Field split calculation for Sentry micro.

Purpose: Compute whether and where to place Force Fields:
         - Ramp block: single FF at ramp bottom when enemy is crossing through
         - Army split: chain of FFs through the enemy center to split them in half
Key Decisions: Ramp block takes priority over split (1 FF at a choke point
               is more impactful than a generic split). Split is always centered
               on the enemy mass — no choke offset. Pooled energy across all
               sentries, greedy assignment.
Limitations: Assumes roughly elliptical enemy formation for splits; L-shaped/
             scattered armies may get suboptimal split lines.
"""

import math
from typing import List, NamedTuple, Optional

import numpy as np
from sc2.game_info import Ramp as Sc2Ramp
from sc2.position import Point2
from sc2.unit import Unit

from cython_extensions import cy_distance_to, cy_find_units_center_mass
from cython_extensions.general_utils import cy_in_pathing_grid_ma

from ares.consts import WORKER_TYPES
from ares.dicts.unit_data import UNIT_DATA

from bot.constants import (
    COMMON_UNIT_IGNORE_TYPES,
    FF_CAST_RANGE,
    FF_ENERGY_COST,
    FF_OVERLAP,
    FF_RADIUS,
    FF_RAMP_BLOCK_MIN_VALUE,
    FF_RAMP_BLOCK_RADIUS,
    FF_SPLIT_MIN_ENEMIES,
)


class FFSplitResult(NamedTuple):
    """Result from compute_ff_split with assignments and debug info."""
    assignments: list[tuple[Unit, Point2]]
    enemy_center: Point2


def _normalize(v: Point2) -> Point2:
    """Return unit-length direction vector, or (1, 0) if zero-length."""
    length = math.sqrt(v.x ** 2 + v.y ** 2)
    if length < 1e-6:
        return Point2((1.0, 0.0))
    return Point2((v.x / length, v.y / length))


def _ramp_center(ramp: Sc2Ramp) -> Point2:
    """Compute the centroid of ALL ramp tiles — the true center of the ramp.

    Unlike top_center/bottom_center (centroids of only upper/lower subsets),
    this accounts for both left-right and top-bottom asymmetry. On asymmetric
    ramps, the upper/lower centroids are pulled toward the wider side.
    """
    pts = ramp.points
    n = len(pts)
    return Point2((
        sum(p.x for p in pts) / n,
        sum(p.y for p in pts) / n,
    ))


def compute_ff_ramp_block(
    enemies: List[Unit],
    sentries: List[Unit],
    own_ramp: Optional[Sc2Ramp],
    enemy_ramp: Optional[Sc2Ramp],
) -> Optional[FFSplitResult]:
    """Check if the enemy center is crossing a ramp and place a single FF to block it.

    Only considers our ramp (enemies pushing in) and the enemy's ramp (enemies
    escaping out). Neutral ramps across the map are ignored — blocking those
    rarely matters and wastes energy.

    Places the FF at the centroid of ALL ramp tiles — this is the true center
    of the ramp regardless of shape, accounting for both left-right and
    top-bottom asymmetry.

    Args:
        enemies: All nearby enemy units (pre-filtered for combat relevance)
        sentries: All friendly sentries (energy pooled across all of them)
        own_ramp: Our main Ramp object (from bot.main_base_ramp), or None
        enemy_ramp: Enemy main Ramp object (from bot.mediator.get_enemy_ramp), or None

    Returns:
        FFSplitResult with a single FF assignment at the ramp center, or None
        if no ramp block opportunity is detected.
    """
    # Filter to ground combat enemies
    ground_enemies = [
        e for e in enemies
        if not e.is_flying
        and e.type_id not in COMMON_UNIT_IGNORE_TYPES
        and e.type_id not in WORKER_TYPES
        and not e.is_structure
    ]

    # Army value gate: is this force worth spending 50 energy on?
    enemy_value = sum(
        UNIT_DATA.get(e.type_id, {}).get("army_value", 0.0)
        for e in ground_enemies
    )
    if enemy_value < FF_RAMP_BLOCK_MIN_VALUE:
        return None

    # Enemy center of mass
    enemy_center_arr, _ = cy_find_units_center_mass(ground_enemies, 20.0)
    enemy_center = Point2(enemy_center_arr)

    # Check if enemy center is near our ramp or their ramp.
    # Use the centroid of ALL ramp tiles (ramp.points) — this is the true
    # center of the ramp regardless of shape, accounting for both left-right
    # and top-bottom asymmetry. top_center/bottom_center are centroids of
    # only the upper/lower subsets, which are offset on asymmetric ramps.
    target_pos: Optional[Point2] = None
    best_dist = FF_RAMP_BLOCK_RADIUS

    if own_ramp is not None:
        ramp_center = _ramp_center(own_ramp)
        dist = cy_distance_to(enemy_center, ramp_center)
        if dist < best_dist:
            best_dist = dist
            target_pos = ramp_center

    if enemy_ramp is not None:
        ramp_center = _ramp_center(enemy_ramp)
        dist = cy_distance_to(enemy_center, ramp_center)
        if dist < best_dist:
            best_dist = dist
            target_pos = ramp_center

    if target_pos is None:
        return None

    # Find closest sentry with enough energy and in cast range
    for s in sentries:
        if s.energy >= FF_ENERGY_COST and cy_distance_to(s.position, target_pos) <= FF_CAST_RANGE:
            return FFSplitResult(
                assignments=[(s, target_pos)],
                enemy_center=enemy_center,
            )

    return None


def compute_ff_split(
    enemies: List[Unit],
    sentries: List[Unit],
    own_center: Point2,
    ground_grid: np.ndarray,
) -> Optional[FFSplitResult]:
    """Determine if a force field split is feasible and return cast assignments.

    Algorithm:
      1. Filter to ground combat enemies (no flying, no workers, no ignores)
      2. Compute enemy center of mass
      3. Determine split axis: direction from enemy center toward our army
      4. Walk along the perpendicular line placing FF positions
      5. Validate edge positions (beyond enemy cluster) are pathable
      6. Pool energy across all sentries; greedy assign each FF to closest
         sentry with enough remaining energy and within cast range

    The split is always centered on the enemy mass — no choke offset.
    Choke/ramp situations are handled by compute_ff_ramp_block() which
    places a single FF at the ramp bottom, which is more effective than
    trying to split through a bottleneck.

    Args:
        enemies: All nearby enemy units (pre-filtered for combat relevance)
        sentries: All friendly sentries (energy pooled across all of them)
        own_center: Our army center of mass (determines split direction)
        ground_grid: Ground grid for pathing validation (cy_in_pathing_grid_ma)

    Returns:
        FFSplitResult with assignments + debug info, or None if split not feasible.
        Assignments is a list of (sentry, target_position) pairs. Each sentry
        may appear multiple times if it has enough energy for multiple FFs.
    """
    # 1. Filter to ground combat enemies only
    ground_enemies = [
        e for e in enemies
        if not e.is_flying
        and e.type_id not in COMMON_UNIT_IGNORE_TYPES
        and e.type_id not in WORKER_TYPES
        and not e.is_structure
    ]
    if len(ground_enemies) < FF_SPLIT_MIN_ENEMIES:
        return None

    # 2. Enemy center of mass
    enemy_center_arr, _ = cy_find_units_center_mass(ground_enemies, 20.0)
    enemy_center = Point2(enemy_center_arr)

    # 3. Split axis: direction from enemy center toward our army
    #    This cuts the army perpendicular to the engagement axis,
    #    separating the front (engaging us) from the back (reinforcing)
    split_dir = _normalize(own_center - enemy_center)
    # Perpendicular direction: the line we place FFs along
    perp_dir = Point2((-split_dir.y, split_dir.x))

    # 4. Project enemies onto perpendicular axis for FF chain extent
    perp_projections: list[float] = []
    for e in ground_enemies:
        offset = e.position - enemy_center
        proj_perp = offset.x * perp_dir.x + offset.y * perp_dir.y
        perp_projections.append(proj_perp)

    # 6. Calculate FF chain along the perpendicular line
    #    Extend slightly beyond the enemy cluster to ensure full coverage
    cluster_min_perp = min(perp_projections)
    cluster_max_perp = max(perp_projections)
    min_perp = cluster_min_perp - FF_RADIUS
    max_perp = cluster_max_perp + FF_RADIUS
    span = max_perp - min_perp

    # Each FF covers 2*FF_RADIUS diameter, with FF_OVERLAP overlap
    ff_spacing = 2 * FF_RADIUS - FF_OVERLAP
    num_ffs = max(1, math.ceil(span / ff_spacing))

    # 7. Generate FF positions along the split line
    #    Positions within the enemy cluster span are pathable by definition
    #    (enemies are standing there). Only validate positions that extend
    #    beyond the cluster, where the line may cross cliffs/water.
    ff_positions: list[Point2] = []
    for i in range(num_ffs):
        t = min_perp + ff_spacing * (i + 0.5)
        pos = Point2((
            enemy_center.x + perp_dir.x * t,
            enemy_center.y + perp_dir.y * t,
        ))
        # Only pathing-check positions outside the enemy cluster span.
        # Inside the cluster, enemies are standing there — it's pathable.
        is_within_cluster = cluster_min_perp <= t <= cluster_max_perp
        if not is_within_cluster and not cy_in_pathing_grid_ma(ground_grid, pos):
            # Position extends beyond the cluster onto potentially unpathable
            # terrain (cliffs, water, walls). Try nudging along the split axis.
            found_valid = False
            for offset_dist in [0.5, -0.5, 1.0, -1.0]:
                alt_pos = Point2((
                    pos.x + split_dir.x * offset_dist,
                    pos.y + split_dir.y * offset_dist,
                ))
                if cy_in_pathing_grid_ma(ground_grid, alt_pos):
                    pos = alt_pos
                    found_valid = True
                    break
            if not found_valid:
                # Can't place this edge FF — skip it rather than waste energy
                continue
        ff_positions.append(pos)

    if not ff_positions:
        return FFSplitResult(assignments=[], enemy_center=enemy_center)

    # 8. Check total pooled energy across all sentries
    total_energy = sum(s.energy for s in sentries)
    if total_energy < len(ff_positions) * FF_ENERGY_COST:
        return FFSplitResult(assignments=[], enemy_center=enemy_center)

    # 9. Greedy assignment: each FF to closest sentry with energy + in range
    assignments: list[tuple[Unit, Point2]] = []
    sentry_energy: dict[int, float] = {s.tag: s.energy for s in sentries}

    for pos in ff_positions:
        best_sentry: Optional[Unit] = None
        best_dist = float("inf")
        for s in sentries:
            if sentry_energy[s.tag] < FF_ENERGY_COST:
                continue
            dist = cy_distance_to(s.position, pos)
            if dist > FF_CAST_RANGE:
                continue
            if dist < best_dist:
                best_dist = dist
                best_sentry = s

        if best_sentry is None:
            # Can't assign this FF — split fails
            return FFSplitResult(assignments=[], enemy_center=enemy_center)

        assignments.append((best_sentry, pos))
        sentry_energy[best_sentry.tag] -= FF_ENERGY_COST

    return FFSplitResult(
        assignments=assignments, enemy_center=enemy_center,
    )
