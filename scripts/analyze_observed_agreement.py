"""Observed-category vs replay-label agreement analysis.

Purpose: Validate classify_observed_game()'s rubric against replay ground truth.
         observed_category is the bot's fog-of-war classification of what happened;
         strategy_category is the server's replay-derived label. Agreement rates
         tell us which OBSERVED_* thresholds need tuning.

Key Decisions: Read-only analysis — fetches from match-level-full, joins the two
               labels, reports agreement by category and by observed_category_source.
               Never feeds back into training features (observed_category is ground
               truth — feature use would be label leakage).

Usage:
    python scripts/analyze_observed_agreement.py [--limit 2000]
"""

import argparse
import collections
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "https://telemetry.pownz.com/api"


def fetch(limit: int) -> list[dict]:
    resp = requests.get(f"{API_BASE}/features/match-level-full?limit={limit}", timeout=90)
    resp.raise_for_status()
    return resp.json()


def norm(label: str | None) -> str | None:
    """Normalize API timing -> timing_attack; drop empties."""
    if not label:
        return None
    label = str(label).strip().lower()
    if label == "timing":
        label = "timing_attack"
    return label if label in ("cheese", "all_in", "timing_attack", "macro") else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Observed vs replay label agreement")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    rows = fetch(args.limit)

    # Join: games with BOTH observed_category and replay label
    both = [
        m for m in rows
        if norm(m.get("observed_category")) and norm(m.get("strategy_category"))
    ]
    print(f"games with both labels: {len(both)} / {len(rows)} fetched")
    if not both:
        print("No overlap yet — deploy and accumulate games first.")
        return 0

    cats = ["cheese", "all_in", "timing_attack", "macro"]

    # ── Overall agreement ──
    agree = sum(1 for m in both
                if norm(m["observed_category"]) == norm(m["strategy_category"]))
    print(f"\n=== OVERALL AGREEMENT: {agree}/{len(both)} ({agree/len(both)*100:.1f}%) ===\n")

    # ── Confusion matrix (observed vs replay) ──
    cm = collections.Counter(
        (norm(m["observed_category"]), norm(m["strategy_category"])) for m in both
    )
    print("Confusion (rows = observed, cols = replay):")
    header = "            " + "".join(f"{c:>13s}" for c in cats)
    print(header)
    for oc in cats:
        line = f"{oc:>12s}" + "".join(
            f"{cm.get((oc, rc), 0):>13d}" for rc in cats
        )
        print(line)

    # ── Per-source agreement (which rules are reliable?) ──
    print("\n=== PER-SOURCE BREAKDOWN ===")
    by_source = collections.defaultdict(list)
    for m in both:
        by_source[m.get("observed_category_source") or "?"].append(m)
    for source, games in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        ok = sum(1 for m in games
                 if norm(m["observed_category"]) == norm(m["strategy_category"]))
        n = len(games)
        print(f"  {source:<38s} {ok}/{n} agree ({ok/n*100:.0f}%)")

    # ── Per-category recall/precision of the rubric ──
    print("\n=== RUBRIC QUALITY (vs replay) ===")
    for oc in cats:
        # recall: of games the rubric called oc, how many match replay
        called = [m for m in both if norm(m["observed_category"]) == oc]
        if not called:
            continue
        hits = sum(1 for m in called if norm(m["strategy_category"]) == oc)
        # disagreement destinations
        misses = collections.Counter(
            norm(m["strategy_category"]) for m in called if norm(m["strategy_category"]) != oc
        )
        print(f"  {oc:>13s}: called {len(called):>4d}, agree {hits:>4d} "
              f"({hits/len(called)*100:.0f}%)"
              + (f" | misses -> {dict(misses)}" if misses else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())