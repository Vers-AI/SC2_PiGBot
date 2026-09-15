"""Intel quality tracking — how fresh and reliable is our enemy information.

Purpose: Analyze freshness of cached enemy army data and compute urgency
         scores for scout dispatch. Uses unit.age from python-sc2 to determine
         staleness, filtering expired ghost units that UnitCacheManager retains.

Key Decisions: Use unit.age for staleness; filter expired ghosts (age >= 30s)
              from UnitCacheManager. When composition belief is enabled, use
              its probability-weighted freshness as single source of truth.

Limitations: UnitCacheManager never removes units, so we must filter by
            MEMORY_EXPIRY_TIME. is_memory is unreliable for UnitCacheManager
            units since snapshots were taken when visible.
"""

from typing import TYPE_CHECKING

from ares.consts import WORKER_TYPES

from bot.constants import (
    MEMORY_EXPIRY_TIME,
    VISIBLE_AGE_THRESHOLD,
    STALENESS_WINDOW,
    FRESH_INTEL_THRESHOLD,
    URGENCY_BUILD_RATE,
    URGENCY_DECAY_RATE,
)

if TYPE_CHECKING:
    from bot.bot import PiG_Bot


def get_enemy_intel_quality(bot: "PiG_Bot") -> dict:
    """Analyze the freshness of our enemy army intel.

    Uses unit.age from python-sc2 to determine how trustworthy our
    combat simulation data is. Filters out expired ghost units that
    UnitCacheManager retains but UnitMemoryManager has already expired.

    Returns:
        dict with keys:
            - has_intel: bool - have we ever seen enemy army?
            - avg_age: float - average age of active enemy unit data (seconds)
            - visible_count: int - units we can currently see (age < 3s)
            - memory_count: int - units from recent memory (3s-30s)
            - expired_count: int - ghost units filtered out (age >= 30s)
            - freshness: float - 0-1 score (1 = all fresh, 0 = all stale)
    """
    all_cached = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES
    ]

    if not all_cached:
        if bot._enemy_army_ever_seen:
            return {
                "has_intel": True,
                "avg_age": float('inf'),
                "visible_count": 0,
                "memory_count": 0,
                "expired_count": 0,
                "freshness": 0.0,
            }
        return {
            "has_intel": False,
            "avg_age": float('inf'),
            "visible_count": 0,
            "memory_count": 0,
            "expired_count": 0,
            "freshness": 0.0,
        }

    active_units = [u for u in all_cached if u.age < MEMORY_EXPIRY_TIME]
    expired_count = len(all_cached) - len(active_units)

    if not active_units:
        return {
            "has_intel": True,
            "avg_age": max(u.age for u in all_cached),
            "visible_count": 0,
            "memory_count": 0,
            "expired_count": expired_count,
            "freshness": 0.0,
        }

    visible = [u for u in active_units if u.age < VISIBLE_AGE_THRESHOLD]
    memory = [u for u in active_units if u.age >= VISIBLE_AGE_THRESHOLD]

    avg_age = sum(u.age for u in active_units) / len(active_units)

    freshness = sum(
        max(0.0, 1.0 - u.age / STALENESS_WINDOW) for u in active_units
    ) / len(active_units)

    return {
        "has_intel": True,
        "avg_age": avg_age,
        "visible_count": len(visible),
        "memory_count": len(memory),
        "expired_count": expired_count,
        "freshness": freshness,
    }


def update_enemy_intel_tracking(bot: "PiG_Bot") -> None:
    """Update enemy intel tracking flags and intel urgency. Call every frame.

    Sets:
        - bot._enemy_army_ever_seen: sticky flag, True once we've seen enemy army
        - bot._last_enemy_army_visible_time: last time we had direct vision
        - bot._intel_urgency: 0-1 score, builds when stale, decays when fresh
    """
    enemy_army = [
        u for u in bot.mediator.get_cached_enemy_army
        if u.type_id not in WORKER_TYPES
    ]

    visible_army = [u for u in enemy_army if u.age < VISIBLE_AGE_THRESHOLD]

    if visible_army:
        bot._enemy_army_ever_seen = True
        bot._last_enemy_army_visible_time = bot.time
        for unit in visible_army:
            bot._enemy_unit_last_seen[unit.tag] = bot.time
    elif enemy_army:
        bot._enemy_army_ever_seen = True

    # When composition belief is enabled, use its probability-weighted
    # freshness instead of the binary age-threshold freshness.
    use_belief = bot.config.get("Belief", {}).get("enable_composition", False)
    if use_belief:
        freshness = bot.belief_state.composition.freshness
    else:
        intel = get_enemy_intel_quality(bot)
        freshness = intel["freshness"]

    if freshness < FRESH_INTEL_THRESHOLD:
        bot._intel_urgency = min(1.0, bot._intel_urgency + URGENCY_BUILD_RATE)
    else:
        bot._intel_urgency = max(0.0, bot._intel_urgency - URGENCY_DECAY_RATE)
        bot._worker_scout_sent_this_stale_period = False


def get_sim_static_defense(bot: "PiG_Bot", max_count: int | None = None) -> list:
    """Ready enemy static defenses (bunkers, cannons, spines, turrets, PFs)
    for tactical combat sims, capped to bound sim cost vs turtled opponents.

    Source is bot.enemy_structures (live + snapshot) — the cached enemy army
    never contains structures (ARES routes them to a separate branch), so the
    `not u.is_structure` filters on cached lists are no-ops for static D.

    is_ready excludes under-construction bunkers (no DPS yet — and a
    bunker-rush bunker in progress is exactly what we want attackable).
    Snapshots are kept: consistent with existing ghost-unit tolerance in the
    main-army sims, and a scouted bunker rarely changes fast.

    Limitations: empty/salvaged bunkers are treated as full-garrison
    (by design — assume the worst). Dead-but-unscouted snapshot bunkers
    suppress attacks until re-scouted.
    """
    from bot.constants import SIM_STATIC_DEFENSE_MAX, STATIC_DEFENSE_TYPES

    if max_count is None:
        max_count = SIM_STATIC_DEFENSE_MAX
    defense: list = [
        s for s in bot.enemy_structures
        if s.type_id in STATIC_DEFENSE_TYPES and s.is_ready
    ]
    return defense[:max_count]


def update_repair_detection(bot: "PiG_Bot") -> None:
    """Detect enemy SCV/MULE repair of Bunkers and Planetary Fortresses.

    The API gives no repair flag for enemies (orders are hidden, no repair
    buff exists), but Terran structures have zero regen — so HP *increase*
    on a damaged enemy Bunker/PF is unambiguous repair. We track HP per
    visible structure and flag repair on any observed increase.

    Only HP-tick confirms repair: adjacent workers near a merely-damaged
    structure don't count (they could be mining at their own base). Once
    confirmed, nearby SCVs/MULEs are tagged as repairers so targeting
    (disruptor filter + score bonus) can prioritize them.

    Sets (created lazily, safe vs any race):
        - bot._repairing_structures: dict[tag -> {"time", "position"}]
        - bot._repairer_tags: set[int]  (SCV/MULE tags, capped 10)

    Call every frame from bot.py. Perf: O(visible Bunker/PF) with an
    O(workers x repairing) pass that is zero for non-Terran games.
    """
    from sc2.ids.unit_typeid import UnitTypeId

    from bot.constants import (
        REPAIRABLE_STATIC_D_TYPES,
        REPAIR_DETECTOR_TTL,
        REPAIRER_PROXIMITY,
    )

    if not hasattr(bot, "_repair_hp_history"):
        bot._repair_hp_history: dict[int, float] = {}
        bot._repairing_structures: dict[int, dict] = {}
        bot._repairer_tags: set[int] = set()

    history: dict = bot._repair_hp_history
    repairing: dict = bot._repairing_structures
    game_time: float = bot.time

    # 1. HP-tick detection on visible repairable structures
    for s in bot.enemy_structures:
        if s.type_id not in REPAIRABLE_STATIC_D_TYPES or not s.is_visible:
            continue
        tag = s.tag
        prev = history.get(tag)
        if prev is not None and s.health > prev:
            # Terran structures never regen — HP gain means repair
            repairing[tag] = {"time": game_time, "position": s.position}
        history[tag] = s.health

    # 2. Expire stale repair flags
    expired = [t for t, info in repairing.items()
               if game_time - info["time"] > REPAIR_DETECTOR_TTL]
    for t in expired:
        del repairing[t]
        history.pop(t, None)

    # 3. Tag repairers: enemy workers near a confirmed repairing structure
    new_tags: set[int] = set()
    if repairing:
        for w in bot.enemy_units:
            if w.type_id not in (UnitTypeId.SCV, UnitTypeId.MULE):
                continue
            for info in repairing.values():
                if w.distance_to(info["position"]) <= REPAIRER_PROXIMITY:
                    new_tags.add(w.tag)
                    break
    bot._repairer_tags = new_tags