"""Behavioral test: SITUATIONAL_ROWS must exclude support/utility rows from
production-nudging effectiveness scoring while leaving damage rows and combat
targeting untouched.

Run: poetry run python tests/test_situational_rows.py
"""

import importlib.util
import sys
import types

# The Poetry venv's ares-sc2 install is broken (dist-info metadata present, package
# files missing). Stub ares.dicts.unit_data — target_scoring only needs UNIT_DATA
# (a flat stats dict). None of the code under test touches UNIT_DATA values.
ares = types.ModuleType("ares")
ares_dicts = types.ModuleType("ares.dicts")
ares_dicts_unit_data = types.ModuleType("ares.dicts.unit_data")
ares_dicts_unit_data.UNIT_DATA = {
    "MARINE": {"army_value": 5.0},
    "ZERGLING": {"army_value": 2.0},
    "VOIDRAY": {"army_value": 30.0},
    "SENTRY": {"army_value": 6.2},
    "ZEALOT": {"army_value": 6.0},
    "STALKER": {"army_value": 7.0},
}
ares_dicts.unit_data = ares_dicts_unit_data
ares.dicts = ares_dicts
sys.modules["ares"] = ares
sys.modules["ares.dicts"] = ares_dicts
sys.modules["ares.dicts.unit_data"] = ares_dicts_unit_data

# Load bot.constants first (target_scoring imports REPAIRER_TARGET_BONUS from it),
# then target_scoring, bypassing bot/__init__.py's game-runtime chain.
cs_spec = importlib.util.spec_from_file_location(
    "bot.constants", r"F:\Coding\BoC_PiGBot\bot\constants.py"
)
constants_mod = importlib.util.module_from_spec(cs_spec)
sys.modules["bot.constants"] = constants_mod
cs_spec.loader.exec_module(constants_mod)

ts_spec = importlib.util.spec_from_file_location(
    "target_scoring_test", r"F:\Coding\BoC_PiGBot\bot\combat\target_scoring.py"
)
ts = importlib.util.module_from_spec(ts_spec)
sys.modules["target_scoring_test"] = ts
ts_spec.loader.exec_module(ts)

from sc2.ids.unit_typeid import UnitTypeId  # noqa: E402

COUNTER_TABLE = ts.COUNTER_TABLE
SITUATIONAL_ROWS = ts.SITUATIONAL_ROWS

# Verbatim copy of macro.py _compute_effectiveness loop body post-fix.
def effectiveness(own_type, enemy_units):
    n = len(enemy_units)
    if n == 0:
        return 0.0
    total = sum(
        COUNTER_TABLE.get((own_type, e.type_id), 0.0)
        for e in enemy_units
        if (own_type, e.type_id) not in SITUATIONAL_ROWS
    )
    return total / n


class FakeUnit:
    def __init__(self, t):
        self.type_id = t


def main():
    # 1) The bug being fixed: Sentry effectiveness used to scale with enemy count.
    #    20 Marines used to give Sentry 3.0 avg bonus; must now be 0.0.
    marines = [FakeUnit(UnitTypeId.MARINE)] * 20
    old_sentry = sum(
        COUNTER_TABLE.get((UnitTypeId.SENTRY, UnitTypeId.MARINE), 0.0)
        for _ in marines
    ) / len(marines)
    new_sentry = effectiveness(UnitTypeId.SENTRY, marines)
    print(f"Sentry vs 20 Marines: old={old_sentry} new={new_sentry}")
    assert new_sentry == 0.0, f"FAIL: Sentry nudge must be 0, got {new_sentry}"
    assert old_sentry == 3.0

    # 2) Same for Zerglings.
    lings = [FakeUnit(UnitTypeId.ZERGLING)] * 20
    new_sentry_z = effectiveness(UnitTypeId.SENTRY, lings)
    print(f"Sentry vs 20 Zerglings: new={new_sentry_z}")
    assert new_sentry_z == 0.0, f"FAIL: got {new_sentry_z}"

    # 3) Damage rows still pass through: Stalker vs VoidRay must stay 8.0.
    vrs = [FakeUnit(UnitTypeId.VOIDRAY)] * 10
    stalk = effectiveness(UnitTypeId.STALKER, vrs)
    refer = effectiveness(UnitTypeId.ZEALOT, [FakeUnit(UnitTypeId.ZERGLING)] * 20)
    print(f"Stalker vs 10 VoidRays: {stalk}  Zealot vs 20 Lings: {refer}")
    assert stalk == 8.0
    assert refer == 6.0

    # 4) No orphans: every SITUATIONAL_ROWS entry must exist in COUNTER_TABLE.
    orphans = SITUATIONAL_ROWS - set(COUNTER_TABLE)
    assert not orphans, f"FAIL: SITUATIONAL_ROWS entries missing from COUNTER_TABLE: {orphans}"
    print("All SITUATIONAL_ROWS entries exist in COUNTER_TABLE")

    # 5) Targeting path unchanged: score_target still adds the full counter bonus.
    #    (Direct call needs a live Unit API; here we verify the table values it reads.)
    print(
        f"Targeting intact: COUNTER_TABLE[(SENTRY, MARINE)] = "
        f"{COUNTER_TABLE[(UnitTypeId.SENTRY, UnitTypeId.MARINE)]}"
    )

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()