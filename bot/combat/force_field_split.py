"""Force Field split calculation for Sentry micro.

Purpose: Compute whether and where to place a chain of Force Fields
         to split the enemy army in half, cutting off their front from their back.
Key Decisions: Pooled energy across all sentries, choke-aware split line,
               pre-cast pathing validation, greedy sentry-to-FF assignment.
Limitations: Assumes roughly elliptical enemy formation; L-shaped/scattered
             armies may get suboptimal split lines.
"""

import math
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from sc2.position import Point2
from sc2.unit import Unit

from cython_extensions import cy_distance_to, cy_find_units_center_mass
from cython_extensions.general_utils import cy_in_pathing_grid_ma

from ares.consts import WORKER_TYPES

from bot.constants import (
    COMMON_UNIT_IGNORE_TYPES,
    FF_CAST_RANGE,
    FF_ENERGY_COST,
    FF_OVERLAP,
    FF_RADIUS,
    FF_SPLIT_CHOKE_SNAP_RANGE,
    FF_SPLIT_MIN_ENEMIES,
    FF_SPLIT_MIN_HALF,
)


class FFSplitResult(NamedTuple):
    """Result from compute_ff_split with assignments and debug info."""
    assignments: list[tuple[Unit, Point2]]
    enemy_center: Point2
    near_count: int
    far_count: int


def _normalize(v: Point2) -> Point2:
    """Return unit-length direction vector, or (1, 0) if zero-length."""
    length = math.sqrt(v.x ** 2 + v.y ** 2)
    if length < 1e-6:
        return Point2((1.0, 0.0))
    return Point2((v.x / length, v.y / length))


def _find_nearest_choke(
    choke_width_map: dict[Point2, float],
    center: Point2,
    max_range: float,
) -> Optional[Point2]:
    """Find the closest choke tile within max_range of center.

    Scans the choke_width_map for the nearest tile. O(n) over choke tiles
    but the map is pre-filtered to narrow chokes only (typically <50 tiles).

    Args:
        choke_width_map: Dict from create_narrow_choke_points()
        center: Position to search around (enemy center)
        max_range: Maximum distance to consider

    Returns:
        Closest choke tile position, or None if none within range.
    """
    best_tile: Optional[Point2] = None
    best_dist_sq = max_range * max_range
    for tile, _width in choke_width_map.items():
        dx = tile.x - center.x
        dy = tile.y - center.y
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_tile = tile
    return best_tile


def compute_ff_split(
    enemies: List[Unit],
    sentries: List[Unit],
    own_center: Point2,
    choke_width_map: dict[Point2, float],
    ground_grid: np.ndarray,
) -> Optional[FFSplitResult]:
    """Determine if a force field split is feasible and return cast assignments.

    Algorithm:
      1. Filter to ground combat enemies (no flying, no workers, no ignores)
      2. Compute enemy center of mass
      3. Determine split axis: direction from enemy center toward our army
      4. Choke-aware: if a choke is nearby, bias the split line toward it
      5. Project enemies onto split axis → near side vs far side
      6. Check both halves have enough units to justify the split
      7. Walk along the perpendicular line placing FF positions
      8. Validate each position is pathable (pre-cast check)
      9. Pool energy across all sentries; greedy assign each FF to closest
         sentry with enough remaining energy and within cast range

    Args:
        enemies: All nearby enemy units (pre-filtered for combat relevance)
        sentries: All friendly sentries (energy pooled across all of them)
        own_center: Our army center of mass (determines split direction)
        choke_width_map: Dict from create_narrow_choke_points() for choke awareness
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

    # 4. Choke-aware: if a choke is nearby, snap the split line toward it
    #    FFs at chokes are far more effective because the enemy can't easily
    #    path around them (constrained terrain on both sides)
    choke_tile = _find_nearest_choke(
        choke_width_map, enemy_center, FF_SPLIT_CHOKE_SNAP_RANGE
    )
    if choke_tile is not None:
        # Bias the split line to pass through the choke
        # Blend the perpendicular direction with the direction toward the choke
        to_choke = _normalize(choke_tile - enemy_center)
        # The split line should pass through the choke, so we adjust the
        # center of the FF chain toward the choke tile
        # Weighted blend: 70% choke direction, 30% original perpendicular
        # This keeps the split roughly perpendicular but shifts it toward the choke
        blended_x = perp_dir.x * 0.3 + to_choke.x * 0.7
        blended_y = perp_dir.y * 0.3 + to_choke.y * 0.7
        # Recompute perpendicular to the blended direction
        blended_dir = _normalize(Point2((blended_x, blended_y)))
        perp_dir = Point2((-blended_dir.y, blended_dir.x))
        # Shift the enemy center toward the choke for FF placement
        enemy_center = Point2((
            enemy_center.x + (choke_tile.x - enemy_center.x) * 0.3,
            enemy_center.y + (choke_tile.y - enemy_center.y) * 0.3,
        ))

    # 5. Project enemies onto split axis to determine near/far halves
    #    "Near" = toward our army (front), "Far" = away (back/reinforcements)
    near_count = 0
    far_count = 0
    perp_projections: list[float] = []
    for e in ground_enemies:
        offset = e.position - enemy_center
        # Along split axis (near vs far)
        proj_along = offset.x * split_dir.x + offset.y * split_dir.y
        if proj_along <= 0:
            near_count += 1
        else:
            far_count += 1
        # Along perpendicular axis (for FF chain extent)
        proj_perp = offset.x * perp_dir.x + offset.y * perp_dir.y
        perp_projections.append(proj_perp)

    # 6. Both halves must have enough units to justify the energy investment
    if near_count < FF_SPLIT_MIN_HALF or far_count < FF_SPLIT_MIN_HALF:
        return FFSplitResult(
            assignments=[], enemy_center=enemy_center,
            near_count=near_count, far_count=far_count,
        )

    # 7. Calculate FF chain along the perpendicular line
    #    Extend slightly beyond the enemy cluster to ensure full coverage
    min_perp = min(perp_projections) - FF_RADIUS
    max_perp = max(perp_projections) + FF_RADIUS
    span = max_perp - min_perp

    # Each FF covers 2*FF_RADIUS diameter, with FF_OVERLAP overlap
    ff_spacing = 2 * FF_RADIUS - FF_OVERLAP  # ~2.5 tiles
    num_ffs = max(1, math.ceil(span / ff_spacing))

    # 8. Generate FF positions along the split line and validate pathing
    ff_positions: list[Point2] = []
    for i in range(num_ffs):
        t = min_perp + ff_spacing * (i + 0.5)
        pos = Point2((
            enemy_center.x + perp_dir.x * t,
            enemy_center.y + perp_dir.y * t,
        ))
        # Pre-cast validation: position must be pathable ground
        if not cy_in_pathing_grid_ma(ground_grid, pos):
            # Try slight offsets to find a valid position nearby
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
                # Can't place this FF — skip it (gap in chain is better than
                # wasting energy on an invalid cast)
                continue
        ff_positions.append(pos)

    if not ff_positions:
        return FFSplitResult(
            assignments=[], enemy_center=enemy_center,
            near_count=near_count, far_count=far_count,
        )

    # 9. Check total pooled energy across all sentries
    total_energy = sum(s.energy for s in sentries)
    if total_energy < len(ff_positions) * FF_ENERGY_COST:
        return FFSplitResult(
            assignments=[], enemy_center=enemy_center,
            near_count=near_count, far_count=far_count,
        )

    # 10. Greedy assignment: each FF to closest sentry with energy + in range
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
            return FFSplitResult(
                assignments=[], enemy_center=enemy_center,
                near_count=near_count, far_count=far_count,
            )

        assignments.append((best_sentry, pos))
        sentry_energy[best_sentry.tag] -= FF_ENERGY_COST

    return FFSplitResult(
        assignments=assignments, enemy_center=enemy_center,
        near_count=near_count, far_count=far_count,
    )
