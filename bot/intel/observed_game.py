"""Observed-game classification — evidence-anchored ground truth from the bot's POV.

Purpose: At game end, record what ACTUALLY happened (from fog-of-war observations)
         instead of what the model predicted. Feeds two consumers:
         1. OpponentBelief runtime profiles (drives cross-game adaptation)
         2. Match-record telemetry (observed_category — comparable vs replay labels)

Key Decisions: Facts only — never reads strategy_label/model output. Priority tree
               mirrors the server labeler's commitment semantics (proximity/unexpanded
               → cheese; early attack → all_in; expanded + mid-window attack →
               timing; long game + expansion + no early attack → macro), drawn from
               bot-observable evidence. Ambiguous games write NOTHING — absence of
               evidence is not evidence of macro (the poison mechanism this replaces).

Limitations: Fog-of-war only — can't see enemy worker counts (the server's economy
             signal). Early attacks without a specific detector record ALL_IN even
             when a replay would call it cheese; both raise the same defensive posture,
             and observed_category_source lets us audit the conflation rate.
"""

from bot.constants import (
    OBSERVED_COMMITTED_GAME_TIME,
    OBSERVED_EARLY_ATTACK_TIME,
    OBSERVED_EARLY_NAT_TIME,
    OBSERVED_LONG_GAME_TIME,
    OBSERVED_NAT_SCOUT_FRESH_TIME,
    StrategyCategory,
)


def classify_observed_game(bot, game_result=None) -> tuple[StrategyCategory, str] | None:
    """Classify the game from observed facts. Returns (category, source) or None.

    Call once at game end. Reads only observations (mediator booleans, guard-rule
    labels, frame-tracked timings, game outcome) — never the model's prediction.
    None means ambiguous: the profile must not be updated for this game.
    """
    first_attack = getattr(bot, "_first_under_attack_time", None)
    nat_start = getattr(bot, "_enemy_nat_started_at", None)
    game_time = bot.time  # at game end this is the final game time
    is_win = game_result is not None and str(game_result).endswith("Victory")

    # ── Rule 1: CHEESE via proximity detectors (highest confidence — structures
    # near OUR base are visible to us by definition) ──
    if bot.reaction_manager.active_reaction_name == "worker_rush":
        return StrategyCategory.CHEESE, "ares:worker_rush"
    if bot.reaction_manager.active_reaction_name == "cannon_rush":
        return StrategyCategory.CHEESE, "ares:cannon_rush"
    if bot.mediator.get_enemy_marine_rush:
        return StrategyCategory.CHEESE, "ares:marine_rush"
    if bot.mediator.get_is_proxy_zealot:
        return StrategyCategory.CHEESE, "ares:proxy_zealot"
    if bot.mediator.get_enemy_four_gate:
        return StrategyCategory.CHEESE, "ares:four_gate"

    # Guard-rule labels (set by strategy_detect when it ran) — also facts
    cheese_label = getattr(bot, "_cheese_label", "none")
    if cheese_label in ("12_pool", "speedling"):
        return StrategyCategory.CHEESE, f"guard:{cheese_label}"
    terran_label = getattr(bot, "_terran_strategy_label", "none")
    if terran_label in ("proxy_rax", "bunker_rush"):
        return StrategyCategory.CHEESE, f"guard:{terran_label}"
    protoss_label = getattr(bot, "_protoss_strategy_label", "none")
    if protoss_label == "proxy_gate":
        return StrategyCategory.CHEESE, "guard:proxy_gateway"

    # Cannon response fired during the game (reaction flag = observed cannons)
    if getattr(bot, "_cannon_rush_response", False) or getattr(bot, "_cannon_rush_active", False):
        return StrategyCategory.CHEESE, "observed:cannons"

    # ── Commitment rules: outcome-conditioned (agreement-validated 2026-09-10,
    # n=483 comparison via analyze_observed_agreement.py). The server's
    # commitment semantics are OUTCOME-based ("no transition if it fails"):
    # an early attack we REPELLED that transitions into a long macro game is
    # pressure within macro, not committed aggression. The bot can't see their
    # economy, but it can observe the outcome: game ended fast + loss = their
    # commitment consumed them (62-68% agreement, vs 8-23% unconditioned).
    # Survived/repelled attacks write nothing — silence beats a wrong label. ──
    if first_attack is not None and first_attack < OBSERVED_EARLY_ATTACK_TIME:
        # Outcome gate: committed aggression shows a fast loss
        if game_time >= OBSERVED_COMMITTED_GAME_TIME or is_win:
            return None  # Attack repelled / game went on — pressure within macro
        # Early attack, fast loss: split cheese vs all_in on the server's
        # unexpanded-commitment line: a FRESH scout showing no nat = observed
        # unexpanded commitment (cheese); no scouting = all_in fallback.
        last_scout = getattr(bot, "_last_nat_scout_time", None)
        nat_present = getattr(bot, "_nat_present_on_last_scout", None)
        if (last_scout is not None
                and nat_present is False
                and first_attack - last_scout <= OBSERVED_NAT_SCOUT_FRESH_TIME):
            return StrategyCategory.CHEESE, "observed:unexpanded_early_attack"
        return StrategyCategory.ALL_IN, "observed:early_attack"

    # NOTE: the old nat_then_midgame_attack timing rule was REMOVED — 0%
    # agreement with replay labels in every conditioning variant (n=122):
    # a nat + mid-game attack describes nearly every ordinary macro game.
    # The server's timing label needs tech-aggression evidence we cannot
    # observe this way. Timing opponents simply write nothing for now.

    if (game_time >= OBSERVED_LONG_GAME_TIME
            and nat_start is not None and nat_start < OBSERVED_EARLY_NAT_TIME
            and (first_attack is None or first_attack >= OBSERVED_EARLY_ATTACK_TIME)):
        # Survived 8+ min, they expanded early, never attacked early —
        # positive evidence of macro (NOT the server's evidence-of-absence,
        # which is invalid under fog of war)
        return StrategyCategory.MACRO, "observed:long_game_early_nat"

    # Ambiguous — write nothing. Dying unsighted to unknown aggression is not
    # a fact about the opponent's strategy, and silence can't poison the profile.
    return None