# scouting

import math
import numpy as np
from typing import Dict, List, Optional, Set
from sc2.data import AbilityId
from sc2.units import Units
from sc2.position import Point2
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import KeepUnitSafe, PathUnitToTarget, UseAbility
from ares.consts import BuildOrderOptions, UnitRole, UnitTreeQueryType, WORKER_TYPES
from bot.combat import attack_target
from bot.constants import FRESH_INTEL_THRESHOLD, MEMORY_EXPIRY_TIME, VISIBLE_AGE_THRESHOLD
from bot.utilities.intel import get_enemy_intel_quality
from cython_extensions import cy_distance_to

# Staleness thresholds for hunt mode
HUNT_URGENCY_THRESHOLD = 0.5  # When _intel_urgency exceeds this, start hunting
LAST_KNOWN_AGE_THRESHOLD = 15.0  # Use last known position if < 15s old


# Radius within which the observer is "near" the enemy army and should orbit
ORBIT_NEAR_RADIUS = 15.0


def get_hunt_target(bot, unit: Unit) -> Point2:
    """
    Get the best target for a unit hunting for the enemy army.
    
    Behaviour priority:
    1. If near the cached army and stale units exist, orbit toward the stalest
       cluster so the observer sweeps the full army instead of sitting at the
       centroid.
    2. If we have recent cached army data (< 15s avg age), head to the centroid.
    3. Fallback: cycle enemy expansions 4th → 3rd → nat → main.
    
    Args:
        bot: The bot instance
        unit: The scouting unit (observer or worker)
    
    Returns:
        Point2: The target position to hunt toward.
    """
    tag = unit.tag
    
    # Get cached enemy army, filtering expired ghosts (age >= MEMORY_EXPIRY_TIME)
    # UnitCacheManager retains units indefinitely but UnitMemoryManager expires at 30s
    cached_army = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES and u.age < MEMORY_EXPIRY_TIME
    ]
    
    if cached_army:
        avg_age = sum(u.age for u in cached_army) / len(cached_army)
        if avg_age < LAST_KNOWN_AGE_THRESHOLD:
            centroid = cached_army[0].position if len(cached_army) == 1 else Point2(
                (sum(u.position.x for u in cached_army) / len(cached_army),
                 sum(u.position.y for u in cached_army) / len(cached_army))
            )
            
            # If already near the army, orbit toward stale units to refresh
            # their visibility instead of parking at the centroid
            if cy_distance_to(unit.position, centroid) < ORBIT_NEAR_RADIUS:
                stale_units = [
                    u for u in cached_army
                    if u.age >= VISIBLE_AGE_THRESHOLD
                ]
                if stale_units:
                    # Pick the stalest unit — this naturally sweeps the
                    # observer across the army footprint
                    stalest = max(stale_units, key=lambda u: u.age)
                    return stalest.position
            
            # Not near the army yet, or every unit is already fresh — head
            # to the centroid
            return centroid
    
    # Fallback: patrol enemy expansions using ARES properties
    # Order: 4th → 3rd → nat → main (armies often stage at outer bases)
    hunt_targets = [
        bot.mediator.get_enemy_fourth,
        bot.mediator.get_enemy_third,
        bot.mediator.get_enemy_nat,
        bot.enemy_start_locations[0],  # Enemy main last
    ]
    
    # Initialize or get current hunt target index
    hunt_key = f"hunt_{tag}"
    if hunt_key not in bot.observer_targets:
        bot.observer_targets[hunt_key] = 0
    
    current_idx = bot.observer_targets[hunt_key]
    current_target = hunt_targets[current_idx % len(hunt_targets)]
    
    # If close to current target, cycle to next
    if cy_distance_to(unit.position, current_target) < 5:
        next_idx = (current_idx + 1) % len(hunt_targets)
        bot.observer_targets[hunt_key] = next_idx
        current_target = hunt_targets[next_idx]
    
    return current_target


def control_worker_scout(bot) -> None:
    """
    Send a worker scout when intel is stale in early game and no observers available.
    
    Triggers when:
    - Early game (game_state == 0)
    - Intel urgency is high (> threshold)
    - No observers available
    """
    # First: check if we have worker scouts that should be recalled
    # (observer available, or no longer early game, or under attack)
    existing_scouts = bot.mediator.get_units_from_role(role=UnitRole.SCOUTING)
    # Filter to only units that still exist by checking their tags
    valid_tags = {u.tag for u in bot.all_own_units}
    existing_scouts = existing_scouts.filter(lambda u: u.tag in valid_tags)
    worker_scouts = existing_scouts.filter(lambda u: u.type_id in WORKER_TYPES)
    
    should_recall = (
        bot.units(UnitTypeId.OBSERVER).amount > 0 or
        bot._under_attack or
        bot._intel_urgency <= HUNT_URGENCY_THRESHOLD  # Intel is fresh, no longer need scout
    )
    
    if worker_scouts and should_recall:
        # Return worker scout to mining
        for scout in worker_scouts:
            # Explicitly clear SCOUTING role first, then assign GATHERING
            bot.mediator.clear_role(tag=scout.tag)
            bot.mediator.assign_role(tag=scout.tag, role=UnitRole.GATHERING)
        # Reset flag so we can send another scout if needed
        bot._worker_scout_sent_this_stale_period = False
        return
    
    # Don't scout if we're under attack - need workers for defense
    if bot._under_attack:
        return
    
    # Only if intel is stale
    if bot._intel_urgency <= HUNT_URGENCY_THRESHOLD:
        return
    
    # Only if we don't have observers
    if bot.units(UnitTypeId.OBSERVER).amount > 0:
        return

    # Don't send worker scout if a hallucinated Phoenix is already scouting
    if bot.units(UnitTypeId.PHOENIX).filter(lambda u: u.is_hallucination):
        return
    
    # Don't send intel scout if build runner already has a worker scout
    build_runner_scouts = bot.mediator.get_units_from_role(role=UnitRole.BUILD_RUNNER_SCOUT)
    build_runner_worker_scouts = build_runner_scouts.filter(lambda u: u.type_id in WORKER_TYPES)
    if build_runner_worker_scouts:
        return
    
    # Check if we already have a worker scout assigned
    existing_scouts = bot.mediator.get_units_from_role(role=UnitRole.SCOUTING)
    # Filter to only units that still exist by checking their tags
    valid_tags = {u.tag for u in bot.all_own_units}
    existing_scouts = existing_scouts.filter(lambda u: u.tag in valid_tags)
    worker_scouts = existing_scouts.filter(lambda u: u.type_id in WORKER_TYPES)
    
    # If flag is set but no worker scouts exist, reset the flag (scout died)
    if bot._worker_scout_sent_this_stale_period and not worker_scouts:
        bot._worker_scout_sent_this_stale_period = False
    
    if worker_scouts:
        # Control existing worker scout
        scout = worker_scouts.first
        actions = CombatManeuver()
        ground_grid = bot.mediator.get_ground_grid
        
        # If damaged, return to base
        if scout.health_percentage < 0.5:
            actions.add(PathUnitToTarget(
                unit=scout,
                target=bot.start_location,
                grid=ground_grid
            ))
            # Explicitly clear SCOUTING role first, then assign GATHERING
            bot.mediator.clear_role(tag=scout.tag)
            bot.mediator.assign_role(tag=scout.tag, role=UnitRole.GATHERING)
        else:
            hunt_target = get_hunt_target(bot, scout)
            actions.add(PathUnitToTarget(
                unit=scout,
                target=hunt_target,
                grid=ground_grid,
                danger_distance=8
            ))
        
        bot.register_behavior(actions)
    else:
        # Need to assign a new worker scout
        worker = bot.mediator.select_worker(target_position=bot.start_location)
        if worker:
            bot.mediator.assign_role(tag=worker.tag, role=UnitRole.SCOUTING)
            bot._worker_scout_sent_this_stale_period = True


# --- Build runner scout safety constants ---
BR_SCOUT_WAYPOINT_DIST = 4.0  # Distance to advance to next waypoint
BR_SCOUT_DANGER_DISTANCE = 10.0  # Danger sensing radius for PathUnitToTarget


def _extract_scout_waypoints(bot) -> list[Point2]:
    """Extract WORKER_SCOUT waypoints from the build order steps."""
    for step in bot.build_order_runner.build_order:
        if (
            step.command == BuildOrderOptions.WORKER_SCOUT
            and isinstance(step.target, list)
            and step.target
        ):
            return list(step.target)
    # Fallback: just go to enemy spawn and back
    return [bot.enemy_start_locations[0]]


def _generate_base_scout_waypoints(bot) -> list[Point2]:
    """Generate waypoints: enemy nat → enemy main perimeter → enemy third.

    Purpose: Hallucinated base_scout follows a logical scouting route — check
    the natural first, then sweep the main base, then the third expansion.
    Key Decisions: Perimeter points sorted by angle for a clean circular path.
    Enemy third chosen over fourth as it's typically closest to the main.
    Limitations: Perimeter shape depends on map; unusual maps may have sparse points.
    """
    enemy_spawn = bot.enemy_start_locations[0]

    try:
        waypoints: list[Point2] = []

        # 1. Enemy natural — check for expand timing
        enemy_nat = bot.mediator.get_enemy_nat
        waypoints.append(enemy_nat)

        # 2. Enemy main — behind mineral line + perimeter sweep
        behind_minerals = bot.mediator.get_behind_mineral_positions(
            th_pos=enemy_spawn
        )
        behind_min_point = behind_minerals[1] if len(behind_minerals) > 1 else None

        map_data = bot.mediator.get_map_data_object
        region = map_data.in_region_p(enemy_spawn)
        perimeter = region.perimeter

        # Filter out points near ramp (already scouted on the way in)
        ramp_top = region.region_ramps[0].top_center.rounded
        ramp_arr = np.array([ramp_top.x, ramp_top.y])
        dists_ramp = np.linalg.norm(perimeter - ramp_arr, axis=1)
        filtered = perimeter[dists_ramp > 8.0]

        # Filter out points near mineral line (covered by behind_min_point)
        if behind_min_point:
            min_arr = np.array([behind_min_point.x, behind_min_point.y])
            dists_min = np.linalg.norm(filtered - min_arr, axis=1)
            filtered = filtered[dists_min > 10.0]

        # Sort perimeter points by angle from enemy spawn for a clean circle
        center = np.array([enemy_spawn.x, enemy_spawn.y])
        if len(filtered) > 0:
            angles = np.arctan2(
                filtered[:, 1] - center[1], filtered[:, 0] - center[0]
            )
            sorted_idx = np.argsort(angles)
            filtered = filtered[sorted_idx]

        n = len(filtered)
        if n > 0:
            if behind_min_point:
                waypoints.append(behind_min_point)
            step = max(1, n // 5)
            indices = [(i * step) % n for i in range(5)]
            for pt in filtered[indices]:
                waypoints.append(Point2(pt))

        # 3. Enemy third — closest expansion to main after natural
        try:
            enemy_third = bot.mediator.get_enemy_expansions[1][0]
            waypoints.append(enemy_third)
        except (IndexError, KeyError):
            pass

        return waypoints if waypoints else [enemy_spawn]

    except Exception:
        # Map data unavailable — simple fallback
        return [enemy_spawn]


def control_build_runner_scout(bot) -> None:
    """
    Take over BUILD_RUNNER_SCOUT workers with safe pathing.

    The ARES build runner fires-and-forgets raw move commands with no danger
    awareness. This function overrides those commands each frame with
    PathUnitToTarget (ground grid + sense_danger) so the scout paths around
    enemy units, and retreats home if damaged or under attack.
    """
    scouts = bot.mediator.get_units_from_role(
        role=UnitRole.BUILD_RUNNER_SCOUT, unit_type=bot.worker_type
    )
    if not scouts:
        # Clean up stale waypoint entries
        if bot._br_scout_waypoints:
            live_tags = {s.tag for s in scouts} if scouts else set()
            bot._br_scout_waypoints = {
                t: v for t, v in bot._br_scout_waypoints.items() if t in live_tags
            }
        return

    ground_grid = bot.mediator.get_ground_grid

    for scout in scouts:
        tag = scout.tag
        actions = CombatManeuver()

        # --- Initialize waypoint tracking for new scouts ---
        if tag not in bot._br_scout_waypoints:
            waypoints = _extract_scout_waypoints(bot)
            bot._br_scout_waypoints[tag] = {"waypoints": waypoints, "idx": 0}

        scout_data = bot._br_scout_waypoints[tag]
        waypoints = scout_data["waypoints"]
        idx = scout_data["idx"]

        # --- All waypoints visited: return to mining ---
        if idx >= len(waypoints):
            bot._br_scout_waypoints.pop(tag, None)
            bot.mediator.clear_role(tag=tag)
            bot.mediator.assign_role(tag=tag, role=UnitRole.GATHERING)
            continue

        target = waypoints[idx]

        # --- Advance to next waypoint when close ---
        if cy_distance_to(scout.position, target) < BR_SCOUT_WAYPOINT_DIST:
            scout_data["idx"] += 1
            if scout_data["idx"] >= len(waypoints):
                bot._br_scout_waypoints.pop(tag, None)
                bot.mediator.clear_role(tag=tag)
                bot.mediator.assign_role(tag=tag, role=UnitRole.GATHERING)
                continue
            target = waypoints[scout_data["idx"]]

        # Priority: flee if shields damaged, then path to waypoint when safe
        if scout.shield_percentage < 1:
            actions.add(KeepUnitSafe(unit=scout, grid=ground_grid))
        actions.add(
            PathUnitToTarget(
                unit=scout,
                target=target,
                grid=ground_grid,
                sense_danger=True,
                danger_distance=BR_SCOUT_DANGER_DISTANCE,
            )
        )
        bot.register_behavior(actions)


def control_observers(bot, all_observers: Units, main_army: Units) -> None:
    """
    Coordinate multiple observers with different roles, adapting to the game situation.
    
    Waterfall priority: army (always filled) → primary (station/siege) → patrol.
    Hunt mode picks the closest observer to the enemy base (preferring non-army).
    
    Parameters:
    - bot: The bot instance
    - all_observers: All observer units (includes OBSERVERSIEGEMODE)
    - main_army: The main army units
    """
    if not all_observers:
        return
    
    # --- Phase 1: Assignment validation (army → primary → patrol) ---
    live_tags = {o.tag for o in all_observers}
    if bot.observer_assignments["army"] and bot.observer_assignments["army"] not in live_tags:
        bot.observer_assignments["army"] = None
    if bot.observer_assignments["primary"] and bot.observer_assignments["primary"] not in live_tags:
        bot.observer_assignments["primary"] = None
    bot.observer_assignments["patrol"] = [
        t for t in bot.observer_assignments["patrol"] if t in live_tags
    ]
    
    assigned_tags: Set[int] = set()
    if bot.observer_assignments["army"]:
        assigned_tags.add(bot.observer_assignments["army"])
    if bot.observer_assignments["primary"]:
        assigned_tags.add(bot.observer_assignments["primary"])
    assigned_tags.update(bot.observer_assignments["patrol"])
    
    # Auto-assign unassigned observers: army first, then primary, then patrol
    for obs in all_observers:
        if obs.tag in assigned_tags:
            continue
        if bot.observer_assignments["army"] is None:
            bot.observer_assignments["army"] = obs.tag
            bot.mediator.clear_role(tag=obs.tag)
            bot.mediator.assign_role(tag=obs.tag, role=UnitRole.CONTROL_GROUP_EIGHT)
        elif bot.observer_assignments["primary"] is None:
            bot.observer_assignments["primary"] = obs.tag
            bot.mediator.clear_role(tag=obs.tag)
            bot.mediator.assign_role(tag=obs.tag, role=UnitRole.SCOUTING)
        else:
            bot.observer_assignments["patrol"].append(obs.tag)
            bot.mediator.clear_role(tag=obs.tag)
            bot.mediator.assign_role(tag=obs.tag, role=UnitRole.CONTROL_GROUP_NINE)
        assigned_tags.add(obs.tag)
    
    # --- Phase 2: Hunt mode resolution ---
    if bot._intel_urgency > HUNT_URGENCY_THRESHOLD:
        bot._observer_hunt_mode = True
    if bot._observer_hunt_mode:
        intel = get_enemy_intel_quality(bot)
        if intel["freshness"] >= FRESH_INTEL_THRESHOLD:
            bot._observer_hunt_mode = False
    
    hunter_tag: Optional[int] = None
    if bot._observer_hunt_mode:
        enemy_pos = bot.enemy_start_locations[0]
        army_tag = bot.observer_assignments["army"]
        
        if len(all_observers) == 1:
            # Only one observer — it hunts (also the army observer)
            hunter_tag = all_observers.first.tag
        else:
            # Prefer closest non-army observer to enemy base
            non_army = [o for o in all_observers if o.tag != army_tag]
            if non_army:
                hunter_tag = min(non_army, key=lambda o: cy_distance_to(o.position, enemy_pos)).tag
            else:
                hunter_tag = min(all_observers, key=lambda o: cy_distance_to(o.position, enemy_pos)).tag
    
    bot._hunting_observer_tag = hunter_tag
    
    # --- Phase 3: Per-observer behavior dispatch ---
    for obs in all_observers:
        if obs.tag == hunter_tag:
            _control_hunting_observer(bot, obs)
        elif obs.tag == bot.observer_assignments["army"]:
            control_army_observer(bot, obs, main_army)
        elif obs.tag == bot.observer_assignments["primary"]:
            control_primary_observer(bot, obs)
        elif obs.tag in bot.observer_assignments["patrol"]:
            _control_single_patrol_observer(bot, obs)
        else:
            # Safety fallback: follow army
            control_army_observer(bot, obs, main_army)


def control_primary_observer(bot, observer: Optional[Unit]) -> None:
    """
    Control the primary observer: station at enemy natural OL spot and siege when arrived.
    Hunt and army-follow are handled by control_observers dispatch.
    
    Parameters:
    - bot: The bot instance
    - observer: The primary observer unit
    """
    if not observer:
        return
    
    # If in siege mode at the station, nothing to do
    if observer.type_id == UnitTypeId.OBSERVERSIEGEMODE:
        return
    
    actions = CombatManeuver()
    air_grid = bot.mediator.get_air_grid
    
    if observer.shield_percentage < 1:
        actions.add(KeepUnitSafe(unit=observer, grid=air_grid))
    else:
        ol_spot = bot.mediator.get_ol_spot_near_enemy_nat
        if ol_spot:
            actions.add(PathUnitToTarget(
                unit=observer,
                target=ol_spot,
                grid=air_grid,
                danger_distance=15,
            ))
            if cy_distance_to(observer.position, ol_spot) < 1:
                observer(AbilityId.MORPH_SURVEILLANCEMODE)
        else:
            actions.add(PathUnitToTarget(
                unit=observer,
                target=bot.enemy_start_locations[0],
                grid=air_grid,
                danger_distance=15,
            ))
    
    bot.register_behavior(actions)


def _control_hunting_observer(bot, observer: Unit) -> None:
    """
    Control an observer in hunt mode: actively seek enemy army.
    If in siege mode, unmorph first to become mobile.
    """
    if observer.type_id == UnitTypeId.OBSERVERSIEGEMODE:
        observer(AbilityId.MORPH_OBSERVERMODE)
        return
    
    actions = CombatManeuver()
    air_grid = bot.mediator.get_air_grid
    
    if observer.shield_percentage < 1:
        actions.add(KeepUnitSafe(unit=observer, grid=air_grid))
    else:
        hunt_target = get_hunt_target(bot, observer)
        actions.add(PathUnitToTarget(
            unit=observer,
            target=hunt_target,
            grid=air_grid,
            danger_distance=12,
        ))
    
    bot.register_behavior(actions)


def _get_forward_army_center(army: Units, target_point: Point2) -> Point2:
    """Get the center of the forward third of army units closest to the attack target.
    
    When the army is spread out, using the full army center puts the observer
    in between the front and back lines. This uses only the forward units as
    the reference point so the observer stays ahead of the push.
    """
    if len(army) <= 3:
        return army.center
    
    # Sort by distance to target, take the forward third (at least 3 units)
    sorted_units = sorted(army, key=lambda u: cy_distance_to(u.position, target_point))
    forward_count = max(3, len(sorted_units) // 3)
    forward_units = sorted_units[:forward_count]
    
    # Average position of forward units
    x = sum(u.position.x for u in forward_units) / forward_count
    y = sum(u.position.y for u in forward_units) / forward_count
    return Point2((x, y))


# Detection search radius around the forward army
_DETECT_SEARCH_RADIUS = 15.0


def control_army_observer(bot, observer: Optional[Unit], main_army: Units) -> None:
    """
    Control the army observer to follow and provide vision for the main army.
    
    Behavior priority:
    1. Flee if taking damage
    2. Move to invisible enemy (is_cloaked or is_burrowed) near army for detection
    3. Provide vision for army blocked by blind ramp (enemies on high ground)
    4. Lead ahead of the forward army units toward the attack target
    
    Parameters:
    - bot: The bot instance
    - observer: The army observer unit
    - main_army: The main army units to follow
    """
    if not observer:
        return
    
    # Army observer should never be in siege mode — unmorph if promoted from
    # a primary that was already sieged (see on_unit_destroyed promotion path).
    if observer.type_id == UnitTypeId.OBSERVERSIEGEMODE:
        observer(AbilityId.MORPH_OBSERVERMODE)
        return
    
    actions = CombatManeuver()
    air_grid = bot.mediator.get_air_grid
    
    if observer.shield_percentage < 1:
        actions.add(KeepUnitSafe(unit=observer, grid=air_grid))
    elif main_army:
        target_point = attack_target(bot, main_army.center)
        forward_center = _get_forward_army_center(main_army, target_point)
        fwd_pos = forward_center
        
        # Find invisible enemies near the forward army.
        # Uses ARES KDTree query around fwd_pos for O(log n) performance,
        # then filters for cloaked/burrowed units that need detection.
        closest_invis = None
        closest_dist = float("inf")
        enemies_near_army = bot.mediator.get_units_in_range(
            start_points=[fwd_pos],
            distances=_DETECT_SEARCH_RADIUS,
            query_tree=UnitTreeQueryType.AllEnemy,
        )[0]
        for enemy in enemies_near_army:
            if not (enemy.is_cloaked or enemy.is_burrowed):
                continue
            if enemy.is_cloaked and enemy.is_revealed:
                continue
            d_fwd = cy_distance_to(enemy.position, fwd_pos)
            if d_fwd < closest_dist:
                closest_dist = d_fwd
                closest_invis = enemy
        
        if closest_invis:
            follow_target = closest_invis.position
        elif bot._blind_ramp_target:
            follow_target = bot._blind_ramp_target
        else:
            # Lead ahead of the forward army toward the attack target
            lead_distance = 10
            tentative = Point2(forward_center.towards(target_point, lead_distance))
            
            try:
                influence = air_grid[int(tentative.x)][int(tentative.y)]
                if influence > 1:
                    lead_distance = 5
            except Exception:
                lead_distance = 8
            
            follow_target = Point2(forward_center.towards(target_point, lead_distance))
        
        # Use the air grid to route around static defense (spores, cannons, turrets).
        # danger_distance=8 gives moderate avoidance without fleeing from the army area.
        actions.add(PathUnitToTarget(
            unit=observer,
            target=follow_target,
            grid=air_grid,
            danger_distance=8,
        ))
    else:
        actions.add(PathUnitToTarget(
            unit=observer,
            target=bot.start_location,
            grid=air_grid,
            danger_distance=15,
        ))
    
    bot.register_behavior(actions)


def _control_single_patrol_observer(bot, observer: Unit) -> None:
    """
    Control a single patrol observer to cycle through key map positions.
    Called per-observer from control_observers dispatch.
    """
    # Patrol observer should never be in siege mode — unmorph first.
    if observer.type_id == UnitTypeId.OBSERVERSIEGEMODE:
        observer(AbilityId.MORPH_OBSERVERMODE)
        return
    
    air_grid = bot.mediator.get_air_grid
    
    ol_spots = bot.mediator.get_ol_spots
    if not ol_spots or len(ol_spots) == 0:
        ol_spots = bot.expansion_locations_list[5:10] if len(bot.expansion_locations_list) > 5 else []
    
    # Filter out the spot near enemy natural (primary observer handles that)
    enemy_nat_spot = bot.mediator.get_ol_spot_near_enemy_nat
    if enemy_nat_spot and enemy_nat_spot in ol_spots:
        ol_spots.remove(enemy_nat_spot)
    
    if not ol_spots:
        return
    
    actions = CombatManeuver()
    tag = observer.tag
    
    if observer.shield_percentage < 1:
        actions.add(KeepUnitSafe(unit=observer, grid=air_grid))
    else:
        # Assign a spot if not already assigned
        if tag not in bot.observer_targets or not bot.observer_targets[tag]:
            # Pick the least-covered spot (simple: hash tag to distribute)
            spot_index = hash(tag) % len(ol_spots)
            bot.observer_targets[tag] = ol_spots[spot_index]
        
        # If close to target, cycle to next spot
        if cy_distance_to(observer.position, bot.observer_targets[tag]) < 1:
            current_index = (
                ol_spots.index(bot.observer_targets[tag])
                if bot.observer_targets[tag] in ol_spots else 0
            )
            next_index = (current_index + 1) % len(ol_spots)
            bot.observer_targets[tag] = ol_spots[next_index]
        
        actions.add(PathUnitToTarget(
            unit=observer,
            target=bot.observer_targets[tag],
            grid=air_grid,
            danger_distance=12,
        ))
    
    bot.register_behavior(actions)


# Legacy function for backward compatibility
def control_scout(bot, scout_units: Units, main_army: Units) -> None:
    """
    Controls your scouting units. This now redirects to the new observer control system.
    """
    # Filter for only observer units
    observers = scout_units.filter(lambda u: u.type_id == UnitTypeId.OBSERVER)
    
    # Use the new observer control system
    control_observers(bot, observers, main_army)


def control_hallucination_scout(bot) -> None:
    """
    Cast hallucinated Phoenix scouts for three scenarios:
    
    1. BLIND RAMP (high priority): Army is blocked at a ramp with no vision,
       and no army observer available. Cast Phoenix to provide high ground vision.
    2. EARLY GAME BASE SCOUT: In early game with no observers, cast Phoenix
       to scout enemy base along the standard probe scout route instead of
       sending a worker. Saves the probe for mining.
    3. INTEL URGENCY (lower priority): Intel is stale and we lack observers.
       Cast Phoenix to scout for the enemy army.
    
    Only one hallucinated scout alive at a time. Cooldown between casts.
    """
    from bot.constants import (
        HALLUCINATION_ENERGY_COST,
        HALLUCINATION_SCOUT_MIN_OBSERVERS,
        HALLUCINATION_SCOUT_COOLDOWN,
        GUARDIAN_SHIELD_ENERGY_COST,
    )

    # Don't cast if a hallucinated scout is still alive
    if bot.units(UnitTypeId.PHOENIX).filter(lambda u: u.is_hallucination):
        return

    # Check if hallucination is researched by seeing if any sentry has the ability
    sentries = bot.units(UnitTypeId.SENTRY).ready
    if not sentries:
        return
    has_hallucination = any(
        AbilityId.HALLUCINATION_PHOENIX in s.abilities
        for s in sentries
    )
    if not has_hallucination:
        return

    # Cooldown check — don't spam hallucination
    if hasattr(bot, '_hallucination_scout_last_cast'):
        if bot.time - bot._hallucination_scout_last_cast < HALLUCINATION_SCOUT_COOLDOWN:
            return

    # Determine if we should cast and what role
    should_cast = False
    scout_role = ""

    # Priority 1: Blind ramp — army needs vision at ramp top, no army observer
    army_observer_tag = bot.observer_assignments.get("army")
    all_observer_tags = (
        {u.tag for u in bot.units(UnitTypeId.OBSERVER)}
        | {u.tag for u in bot.units(UnitTypeId.OBSERVERSIEGEMODE)}
    )
    army_observer_exists = (
        army_observer_tag is not None
        and army_observer_tag in all_observer_tags
    )
    if bot._blind_ramp_target and not army_observer_exists:
        should_cast = True
        scout_role = "high_ground"

    # Priority 2: Early game base scout — replace worker scout with hallucinated Phoenix
    if not should_cast:
        observer_count = (
            bot.units(UnitTypeId.OBSERVER).amount
            + bot.units(UnitTypeId.OBSERVERSIEGEMODE).amount
        )
        br_scouts = bot.mediator.get_units_from_role(
            role=UnitRole.BUILD_RUNNER_SCOUT, unit_type=bot.worker_type
        )
        if (
            bot.game_state == 0
            and bot._intel_urgency > HUNT_URGENCY_THRESHOLD
            and observer_count == 0
            and not br_scouts
        ):
            should_cast = True
            scout_role = "base_scout"

    # Priority 3: Intel urgency — stale intel and not enough observers
    if not should_cast:
        observer_count = (
            bot.units(UnitTypeId.OBSERVER).amount
            + bot.units(UnitTypeId.OBSERVERSIEGEMODE).amount
        )
        if bot._intel_urgency > HUNT_URGENCY_THRESHOLD and observer_count < HALLUCINATION_SCOUT_MIN_OBSERVERS:
            should_cast = True
            scout_role = "intel"

    if not should_cast:
        return

    # Find a sentry with enough energy (reserve energy for Guardian Shield)
    available_sentries = [
        s for s in sentries
        if s.energy >= HALLUCINATION_ENERGY_COST + GUARDIAN_SHIELD_ENERGY_COST
    ]
    if not available_sentries:
        # Try sentries with just enough for hallucination (no shield reserve)
        available_sentries = [s for s in sentries if s.energy >= HALLUCINATION_ENERGY_COST]
    if not available_sentries:
        return

    # Pick the sentry with the most energy (least likely to need it for shield)
    caster = max(available_sentries, key=lambda s: s.energy)

    # Cast hallucinated Phoenix
    caster(AbilityId.HALLUCINATION_PHOENIX)
    bot._hallucination_scout_last_cast = bot.time
    # Role will be assigned in on_unit_created and control_hallucination_scouts
    # Store the intended role so on_unit_created can pick it up
    # (tag not known yet — will be matched in control_hallucination_scouts)
    bot._halu_scout_pending_role = scout_role


def control_hallucination_scouts(bot) -> None:
    """
    Control hallucinated Phoenix scouts — path them based on their role.
    
    Roles:
    - high_ground: Hold position at ramp top for persistent army vision
    - base_scout: Follow standard probe scout waypoints (enemy_spawn → nat → etc.)
    - intel: Hunt for enemy army using get_hunt_target()
    
    Pure move — no attack, no danger avoidance (hallucinations are expendable).
    """
    air_grid = bot.mediator.get_air_grid

    # Find our hallucinated Phoenixes
    hallu_phoenixes = bot.units(UnitTypeId.PHOENIX).filter(
        lambda u: u.is_hallucination
    )

    # Assign roles to new Phoenixes that don't have one yet
    pending_role = getattr(bot, '_halu_scout_pending_role', '')
    for phoenix in hallu_phoenixes:
        if phoenix.tag not in bot._halu_scout_roles:
            # New Phoenix — assign the pending role (set when cast)
            bot._halu_scout_roles[phoenix.tag] = pending_role
            # Also assign the correct ARES UnitRole
            if pending_role == "high_ground":
                bot.mediator.assign_role(tag=phoenix.tag, role=UnitRole.HIGH_GROUND_SPOTTER)
            else:
                bot.mediator.assign_role(tag=phoenix.tag, role=UnitRole.SCOUTING)

    # Clean up stale role entries for dead Phoenixes
    live_tags = {p.tag for p in hallu_phoenixes}
    bot._halu_scout_roles = {
        t: r for t, r in bot._halu_scout_roles.items() if t in live_tags
    }
    bot._halu_scout_waypoints = {
        t: v for t, v in bot._halu_scout_waypoints.items() if t in live_tags
    }

    for phoenix in hallu_phoenixes:
        role = bot._halu_scout_roles.get(phoenix.tag, 'intel')

        if role == "high_ground":
            # Use live ramp target if available, otherwise keep last known
            tag = phoenix.tag
            if bot._blind_ramp_target:
                bot._halu_scout_waypoints[tag] = {
                    "stored_target": bot._blind_ramp_target,
                    "orbit_angle": bot._halu_scout_waypoints.get(
                        tag, {}
                    ).get("orbit_angle", 0.0),
                }
            wp_data = bot._halu_scout_waypoints.get(tag, {})
            center = wp_data.get("stored_target")
            if not center:
                center = bot.start_location

            # Orbit in a small circle (radius 3) around the ramp top
            orbit_radius = 3.0
            angle = wp_data.get("orbit_angle", 0.0)
            target = Point2((
                center.x + orbit_radius * math.cos(angle),
                center.y + orbit_radius * math.sin(angle),
            ))
            # Advance angle when close to current orbit point
            if cy_distance_to(phoenix.position, target) < 1.5:
                wp_data["orbit_angle"] = (angle + 1.0) % (2 * math.pi)
                if tag in bot._halu_scout_waypoints:
                    bot._halu_scout_waypoints[tag] = wp_data

            actions = CombatManeuver()
            actions.add(PathUnitToTarget(
                unit=phoenix,
                target=target,
                grid=air_grid,
                sense_danger=False,
            ))
            bot.register_behavior(actions)

        elif role == "base_scout":
            # Circle the enemy main base, then switch to intel hunting.
            tag = phoenix.tag
            if tag not in bot._halu_scout_waypoints:
                waypoints = _generate_base_scout_waypoints(bot)
                bot._halu_scout_waypoints[tag] = {
                    "waypoints": waypoints, "idx": 0
                }

            wp_data = bot._halu_scout_waypoints[tag]
            waypoints = wp_data.get("waypoints", [])
            idx = wp_data.get("idx", 0)

            if idx < len(waypoints):
                target = waypoints[idx]
                # Advance to next waypoint when close
                if cy_distance_to(phoenix.position, target) < 1:
                    wp_data["idx"] += 1
                    if wp_data["idx"] < len(waypoints):
                        target = waypoints[wp_data["idx"]]
                    else:
                        # Done with waypoints — switch to intel hunting
                        bot._halu_scout_roles[tag] = "intel"
                        target = get_hunt_target(bot, phoenix)
                actions = CombatManeuver()
                actions.add(PathUnitToTarget(
                    unit=phoenix,
                    target=target,
                    grid=air_grid,
                    sense_danger=False,
                ))
                bot.register_behavior(actions)
            else:
                # All waypoints visited — behave as intel scout
                bot._halu_scout_roles[tag] = "intel"
                target = get_hunt_target(bot, phoenix)
                actions = CombatManeuver()
                actions.add(PathUnitToTarget(
                    unit=phoenix,
                    target=target,
                    grid=air_grid,
                    sense_danger=False,
                ))
                bot.register_behavior(actions)

        else:
            # Intel scout — hunt for enemy army
            target = get_hunt_target(bot, phoenix)
            actions = CombatManeuver()
            actions.add(PathUnitToTarget(
                unit=phoenix,
                target=target,
                grid=air_grid,
                sense_danger=False,
            ))
            bot.register_behavior(actions)
