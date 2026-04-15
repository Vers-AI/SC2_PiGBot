"""Weighted target scoring system for combat targeting.

Purpose: Replace simple "attack closest/lowest HP" with multi-factor scoring
Key Decisions: Layered scoring (distance + health + type value + range + counter matchup + threat),
    uses Cython extensions for all heavy math, returns best target per unit
Limitations: O(n*m) where n=own units, m=enemies in radius; mitigated by pre-filtering with cy_closer_than
"""

from typing import Union

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.unit import Unit
from sc2.units import Units

from ares.dicts.unit_data import UNIT_DATA

from cython_extensions import (
    cy_distance_to,
    cy_in_attack_range,
    cy_range_vs_target,
)

# ===== SCORING WEIGHTS =====
# Tuning guide: watch replays, find bad targeting, adjust the weight that should have differentiated.
# Even 1-2 points can change targeting at the margins. Start conservative, iterate from replays.

DISTANCE_WEIGHT = 1.5
"""Per-tile penalty for distance. Higher = prefer closer targets more strongly."""

HEALTH_WEIGHT = 30.0
"""Bonus for low-HP targets. (1 - health_ratio) * HEALTH_WEIGHT → max 30 for nearly dead unit."""

IN_RANGE_BONUS = 15.0
"""Flat bonus when target is already in weapon range (no need to close distance)."""

THREAT_WEIGHT = 5.0
"""Bonus for enemies that can shoot back at us (prefer eliminating threats)."""

# ===== UPGRADE CACHE =====
# Updated once per frame via update_upgrades(). Avoids passing bot into score_target.
_cached_upgrades: set = set()


# ===== UNIT TYPE VALUE (hybrid: UNIT_DATA army_value + tactical bonus) =====
# Base value comes from ARES UNIT_DATA["army_value"] — covers all ~150 unit types automatically.
# TACTICAL_BONUS adds extra points for units whose tactical impact exceeds their economic cost
# (spellcasters, siege, enablers). Only needs entries where army_value under-represents danger.

TYPE_VALUE_SCALE = 0.5
"""Multiplier to bring army_value into our scoring range. army_value ranges 1-101, 
scaled to ~0.5-50. Keeps it balanced with other scoring layers."""

DEFAULT_TYPE_VALUE = 5.0
"""Fallback for unit types not in UNIT_DATA (edge cases, new units)."""

TACTICAL_BONUS: dict[UnitTypeId, float] = {
    # Spellcasters: cheap but game-ending abilities
    UnitTypeId.HIGHTEMPLAR:       25.0,  # Storm devastates clumps (army_value only 8.2)
    UnitTypeId.GHOST:             20.0,  # EMP strips shields / snipe (army_value 11.7)
    UnitTypeId.INFESTOR:          20.0,  # Fungal / neural parasite (army_value 10.4)
    UnitTypeId.INFESTORBURROWED:  20.0,
    UnitTypeId.VIPER:             18.0,  # Abduct / blinding cloud (army_value 18.7)
    UnitTypeId.SENTRY:            15.0,  # Guardian shield / forcefields (army_value 6.2)
    UnitTypeId.ORACLE:            10.0,  # Revelation / harass (army_value 19.0)
    UnitTypeId.RAVEN:             10.0,  # Detection + abilities (army_value 12.4)

    # Siege / zone denial: positional threat undervalued by cost
    UnitTypeId.SIEGETANKSIEGED:   12.0,  # Devastating in position (army_value 17.5)
    UnitTypeId.LURKERMPBURROWED:  10.0,  # Active siege threat (army_value 19.0)
    UnitTypeId.LIBERATORAG:        8.0,  # Zone denial mode (army_value 19.0)
    UnitTypeId.WIDOWMINEBURROWED: 10.0,  # Hidden splash threat (army_value 4.3)
    UnitTypeId.WIDOWMINE:          8.0,  # Pre-burrow still dangerous
    UnitTypeId.BANELINGBURROWED:   8.0,  # Hidden splash suicide (army_value 2.0)
    UnitTypeId.BANELING:           6.0,  # Splash suicide (army_value 2.0)
    UnitTypeId.DISRUPTOR:         12.0,  # Nova one-shots clumps (army_value 19.0)

    # Enablers: removal cripples the army
    UnitTypeId.MEDIVAC:           10.0,  # Healing enabler (army_value 8.4)
    UnitTypeId.WARPPRISM:         12.0,  # Mobility enabler (army_value 8.9)
    UnitTypeId.WARPPRISMPHASING:  12.0,
}

# ===== COUNTER TABLE =====
# (our_type, enemy_type) -> bonus score.
#
# DUAL PURPOSE: This table drives both combat TARGETING (which enemy to shoot)
# and PRODUCTION NUDGING (which of our unit types to build more of).
# When adding a new unit type to army compositions, add its counter entries here
# so the production system knows when to favor it.
#
# HOW TO EXTEND:
#   1. Add (OUR_TYPE, ENEMY_TYPE): bonus entries below
#   2. If the unit type is in an army composition dict (macro.py), production
#      nudging will automatically pick it up — no other changes needed
#   3. Bonus range: 5-12 typical. Higher = stronger counter signal.
#   4. Only add entries where there's a real matchup advantage. Sparse is fine.
#
# DATA SOURCE: Liquipedia SC2 unit pages (Legacy of the Void versions)
#   https://liquipedia.net/starcraft2/<UnitName>_(Legacy_of_the_Void)
#   Check the "Competitive Usage" sections for matchup-specific info.
#   Each entry below is annotated with "# LP:" citing the Liquipedia reasoning.
#
COUNTER_TABLE: dict[tuple[UnitTypeId, UnitTypeId], float] = {
    # Stalkers: anti-air specialist + armored bonus (Liquipedia: "ability to counter air units")
    (UnitTypeId.STALKER, UnitTypeId.COLOSSUS):       10.0,  # Armored tag, high-value snipe
    (UnitTypeId.STALKER, UnitTypeId.IMMORTAL):        5.0,  # Armored tag (targeting only; Immortals hard-counter Stalkers)
    (UnitTypeId.STALKER, UnitTypeId.VOIDRAY):         8.0,  # Anti-air
    (UnitTypeId.STALKER, UnitTypeId.MEDIVAC):         8.0,  # Anti-air, removes healing
    (UnitTypeId.STALKER, UnitTypeId.BANSHEE):         8.0,  # Anti-air
    (UnitTypeId.STALKER, UnitTypeId.MUTALISK):        8.0,  # Anti-air, Blink helps chase
    (UnitTypeId.STALKER, UnitTypeId.WARPPRISM):      10.0,  # Anti-air, high-value snipe
    (UnitTypeId.STALKER, UnitTypeId.LIBERATOR):        8.0,  # Anti-air, Blink past defender zones
    (UnitTypeId.STALKER, UnitTypeId.LIBERATORAG):      8.0,  # Anti-air
    (UnitTypeId.STALKER, UnitTypeId.ORACLE):           8.0,  # Anti-air
    (UnitTypeId.STALKER, UnitTypeId.PHOENIX):          6.0,  # Anti-air
    (UnitTypeId.STALKER, UnitTypeId.CORRUPTOR):        8.0,  # Anti-air (LP: key vs Zerg air)
    (UnitTypeId.STALKER, UnitTypeId.BROODLORD):       8.0,  # LP: "anti Broodlord units"

    # Immortals: anti-armor specialist (LP: "50 base damage against armored")
    (UnitTypeId.IMMORTAL, UnitTypeId.ROACH):         10.0,  # LP: "strongly cost-effective against Roaches"
    (UnitTypeId.IMMORTAL, UnitTypeId.RAVAGER):        8.0,  # Armored tag
    (UnitTypeId.IMMORTAL, UnitTypeId.STALKER):        8.0,  # LP: "deal a large amount of damage to Stalkers"
    (UnitTypeId.IMMORTAL, UnitTypeId.MARAUDER):       8.0,  # LP: "do quite well against...Marauders"
    (UnitTypeId.IMMORTAL, UnitTypeId.SIEGETANK):     10.0,  # LP: "incredibly effective against...Tanks"
    (UnitTypeId.IMMORTAL, UnitTypeId.SIEGETANKSIEGED): 10.0,
    (UnitTypeId.IMMORTAL, UnitTypeId.ULTRALISK):     12.0,  # LP: "shut down Ultralisk transitions"
    (UnitTypeId.IMMORTAL, UnitTypeId.THOR):          10.0,  # LP: "effective against Thors"
    (UnitTypeId.IMMORTAL, UnitTypeId.LURKERMPBURROWED): 10.0,  # LP: "especially key targets like burrowed Lurkers"

    # Zealots: melee + Charge vs light/slow (only with Charge — see score_target)
    (UnitTypeId.ZEALOT, UnitTypeId.MARINE):           8.0,  # LP: Charge closes distance to ranged Terran
    (UnitTypeId.ZEALOT, UnitTypeId.HYDRALISK):        6.0,  # LP: "+1 kills Hydra in 5 hits instead of 6"
    (UnitTypeId.ZEALOT, UnitTypeId.ZERGLING):         6.0,  # LP: "very effective...with even surface areas"
    (UnitTypeId.ZEALOT, UnitTypeId.SIEGETANKSIEGED):  8.0,  # LP: "effective threat against Siege Tanks...absorb 4 shots...friendly-fire"
    (UnitTypeId.ZEALOT, UnitTypeId.MARAUDER):         6.0,  # LP: "competitive against...particularly the Marauder"
    # NOTE: Zealot vs Baneling REMOVED — LP: "Banelings very effective against massed Zealots"

    # Archons: anti-light / anti-bio splash + high shields
    (UnitTypeId.ARCHON, UnitTypeId.MUTALISK):        12.0,  # Premier anti-Muta unit
    (UnitTypeId.ARCHON, UnitTypeId.ZERGLING):         8.0,  # Splash massacres lings
    (UnitTypeId.ARCHON, UnitTypeId.MARINE):           8.0,  # Splash vs bio
    (UnitTypeId.ARCHON, UnitTypeId.HYDRALISK):        8.0,  # Splash effective
    (UnitTypeId.ARCHON, UnitTypeId.BANELING):        10.0,  # High shields absorb baneling hits

    # Colossus: splash vs light units (LP: "bonus damage against light")
    (UnitTypeId.COLOSSUS, UnitTypeId.MARINE):        10.0,  # LP: "good against Marine-heavy...bonus against Light"
    (UnitTypeId.COLOSSUS, UnitTypeId.ZERGLING):      10.0,  # LP: light unit, splash
    (UnitTypeId.COLOSSUS, UnitTypeId.HYDRALISK):      8.0,  # LP: light unit, splash
    (UnitTypeId.COLOSSUS, UnitTypeId.ZEALOT):         6.0,  # LP: "very strong against large numbers of Zealots"
    (UnitTypeId.COLOSSUS, UnitTypeId.ADEPT):          6.0,  # Light unit, splash
    (UnitTypeId.COLOSSUS, UnitTypeId.BANELING):       8.0,  # Light unit, low HP — splash destroys
    # NOTE: Colossus vs Roach REMOVED — LP: "not as cost-effective" (no armor bonus)
    # Colossus HARD-COUNTERED by anti-air: negative values reduce production via nudging
    (UnitTypeId.COLOSSUS, UnitTypeId.VIKINGFIGHTER):  -10.0,  # LP: "Vikings...primary counter to Colossi"
    (UnitTypeId.COLOSSUS, UnitTypeId.CORRUPTOR):      -10.0,  # LP: "Corruptors are the primary Zerg counter to Colossi"
    (UnitTypeId.COLOSSUS, UnitTypeId.PHOENIX):        -8.0,   # PvP: Phoenix lifts Colossus, hard counter
    (UnitTypeId.COLOSSUS, UnitTypeId.CARRIER):        -8.0,   # PvP: Carrier outranges, Colossus can't shoot air
    (UnitTypeId.COLOSSUS, UnitTypeId.TEMPEST):       -10.0,   # PvP: Tempest outranges everything, Colossus helpless

    # Disruptor: burst splash vs immobile/clumped armies
    (UnitTypeId.DISRUPTOR, UnitTypeId.SIEGETANKSIEGED): 10.0,  # LP: "very good against units that cannot move"
    (UnitTypeId.DISRUPTOR, UnitTypeId.SIEGETANK):      8.0,
    (UnitTypeId.DISRUPTOR, UnitTypeId.MARINE):         8.0,  # Nova one-shots
    (UnitTypeId.DISRUPTOR, UnitTypeId.MARAUDER):       8.0,  # LP: "effective response to...Marauders"
    (UnitTypeId.DISRUPTOR, UnitTypeId.ROACH):          6.0,  # LP: "Roaches...clumps is very vulnerable"
    (UnitTypeId.DISRUPTOR, UnitTypeId.RAVAGER):        6.0,  # LP: "Ravagers...clumps is very vulnerable"
    (UnitTypeId.DISRUPTOR, UnitTypeId.HYDRALISK):      6.0,  # Clumped hydras vulnerable
    (UnitTypeId.DISRUPTOR, UnitTypeId.IMMORTAL):       5.0,  # Doesn't one-shot (200HP+100S), but decent
    (UnitTypeId.DISRUPTOR, UnitTypeId.LURKERMPBURROWED): 8.0,  # LP: "Lurkers" can't move
    # NOTE: Disruptor vs Stalker REMOVED — LP: "Stalkers can use Blink to avoid Nova"

    # High Templar: Psionic Storm vs clumped armies + Feedback vs casters
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.MARINE):      10.0,  # LP: "devastated with properly placed Storms"
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.MARAUDER):     8.0,  # LP: "large Marine/Marauder groups"
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.HYDRALISK):   10.0,  # LP: "storm will lower Hydras...Immortals clean up"
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.ZERGLING):     8.0,  # LP: Storm kills Zerglings outright
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.BANELING):     8.0,  # Storm kills Banelings outright (35 HP)
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.ROACH):        6.0,  # Clumped Roaches take storm damage
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.MUTALISK):     4.0,  # LP: "Mutas escape storms fairly easily...Archons fare better"
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.GHOST):       10.0,  # LP: "Feedback...counteracted by using Feedback on Ghost"
    (UnitTypeId.HIGHTEMPLAR, UnitTypeId.VIPER):       10.0,  # LP: "Feedback is only really useful versus Vipers"
}


def update_upgrades(upgrades: set) -> None:
    """Cache the bot's researched upgrades for use in score_target.

    Call once per frame from combat.py before any scoring happens.
    """
    global _cached_upgrades
    _cached_upgrades = upgrades


def score_target(my_unit: Unit, enemy: Unit) -> float:
    """Score a single enemy target from the perspective of my_unit.

    Combines base score (universal enemy value) with per-unit score (distance,
    range, effectiveness). Higher score = better target.

    Perf note: Uses Cython cy_distance_to (~157ns) and cy_range_vs_target for
    all heavy math. One call per (my_unit, enemy) pair.

    Args:
        my_unit: Our unit evaluating the target.
        enemy: The enemy unit being scored.

    Returns:
        Combined score (float). Higher is better.
    """
    score = 0.0

    # --- Base Score (same for all own units) ---

    # Layer 3: Unit type value (army_value base + tactical bonus)
    data = UNIT_DATA.get(enemy.type_id)
    base_value = (data["army_value"] * TYPE_VALUE_SCALE) if data else DEFAULT_TYPE_VALUE
    score += base_value + TACTICAL_BONUS.get(enemy.type_id, 0.0)

    # Layer 2: Health — lower HP = higher priority (focus fire)
    if enemy.health_max > 0:
        health_ratio = (enemy.health + enemy.shield) / (enemy.health_max + enemy.shield_max)
        score += (1.0 - health_ratio) * HEALTH_WEIGHT

    # Counter table bonus (Zealots only get counter bonuses with Charge upgrade)
    if my_unit.type_id == UnitTypeId.ZEALOT:
        if UpgradeId.CHARGE in _cached_upgrades:
            score += COUNTER_TABLE.get((my_unit.type_id, enemy.type_id), 0.0)
    else:
        score += COUNTER_TABLE.get((my_unit.type_id, enemy.type_id), 0.0)

    # --- Per-Unit Score (specific to this unit) ---

    # Layer 1: Distance — closer = better
    dist = cy_distance_to(my_unit.position, enemy.position)
    score -= dist * DISTANCE_WEIGHT

    # In-range bonus — no movement needed to shoot
    weapon_range = cy_range_vs_target(my_unit, enemy)
    if dist <= weapon_range:
        score += IN_RANGE_BONUS

    # Threat: enemies that can shoot us back are higher priority
    enemy_range = cy_range_vs_target(enemy, my_unit)
    if enemy_range > 0 and dist <= enemy_range + 2.0:
        score += THREAT_WEIGHT

    return score


def select_target(
    my_unit: Unit,
    enemies: Union[Units, list[Unit]],
) -> Unit:
    """Pick the best target for my_unit from a list of enemies using weighted scoring.

    Each enemy is scored by score_target(). Returns the highest-scored enemy.
    This replaces cy_pick_enemy_target for combat targeting decisions.

    Perf note: O(m) per call where m = len(enemies). Pre-filter enemies with
    cy_closer_than or get_units_in_range before calling.

    Args:
        my_unit: Our unit choosing a target.
        enemies: Candidate enemy units (pre-filtered to attackable/reachable).

    Returns:
        The best enemy Unit to target. Returns first enemy if list has one element.
    """
    if len(enemies) == 1:
        return enemies[0]

    best_target = enemies[0]
    best_score = score_target(my_unit, enemies[0])

    for enemy in enemies[1:]:
        s = score_target(my_unit, enemy)
        if s > best_score:
            best_score = s
            best_target = enemy

    return best_target
