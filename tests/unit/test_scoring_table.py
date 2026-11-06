"""Format + coverage validation for SCORING_TABLE (bot/combat/scoring_table.py).

Purpose: pin the table's invariants so a bad row cannot slip in unnoticed —
    duplicate pairs, keys that do not resolve, off-scale values, coverage holes.
Key Decisions: the table is inspected directly — no game, no SC2 binary. The
    duplicate check parses the source with `ast`: a Python dict keeps only the
    last value for a repeated key, so the runtime table cannot show duplicates.
    The composition roster is read from macro.py the same way, so the check
    never imports a manager (and stays pure Python).
Limitations: the coverage rule is structural — it cannot tell whether a row is
    strategically right. The Liquipedia citations behind each row live in
    plan/counter_table_research_liquipedia.md.
"""

import ast
from pathlib import Path

from sc2.ids.unit_typeid import UnitTypeId

from bot.combat.scoring_table import SCORING_TABLE, SITUATIONAL_ROWS

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_SOURCE = REPO_ROOT / "bot" / "combat" / "scoring_table.py"
MACRO_SOURCE = REPO_ROOT / "bot" / "managers" / "macro.py"

SCORE_MIN = -12.0
"""Lower bound of the table's value scale. The observed range is -10..12; the
extra step of headroom makes a typo (120.0) fail instead of silently scoring."""

SCORE_MAX = 12.0
"""Upper bound of the table's value scale (see SCORE_MIN)."""


# ===== SOURCE PARSING HELPERS =====
# The runtime table hides duplicates (dict literal semantics), so the source is
# the ground truth for "which pairs are listed".

def _assigned_name(node: ast.stmt) -> str | None:
    """Name of the single module-level variable a statement assigns, if any."""
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = node.targets
    else:
        return None
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def _unit_type_pairs(literal_name: str, source: Path) -> list[tuple[str, str]]:
    """Return the (our_type, enemy_type) member names of a table literal.

    Args:
        literal_name: Name of the module-level literal to read ("SCORING_TABLE").
        source: Python file holding the literal.

    Returns:
        List of ("UnitTypeId.STALKER", "UnitTypeId.COLOSSUS") name pairs, in
        source order. Each member must be an `UnitTypeId.X` attribute — anything
        else raises AssertionError.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if _assigned_name(node) != literal_name:
            continue
        value = node.value
        elements = value.keys if isinstance(value, ast.Dict) else value.elts
        pairs = []
        for element in elements:
            assert isinstance(element, ast.Tuple) and len(element.elts) == 2, (
                f"{literal_name} entries must be (our_type, enemy_type) tuples"
            )
            members = []
            for member in element.elts:
                assert isinstance(member, ast.Attribute) and isinstance(member.value, ast.Name), (
                    f"{literal_name} entries must reference UnitTypeId members"
                )
                members.append(f"{member.value.id}.{member.attr}")
            pairs.append((members[0], members[1]))
        return pairs
    raise AssertionError(f"{literal_name} literal not found in {source}")


def _composition_unit_types() -> set[str]:
    """Unit type names the bot puts into its army compositions (macro.py).

    Parsed, not imported: the roster must not drag the manager's runtime
    dependencies (model files, ARES) into a pure-Python unit test.
    """
    tree = ast.parse(MACRO_SOURCE.read_text(encoding="utf-8"))
    roster: set[str] = set()
    for node in ast.walk(tree):
        name = _assigned_name(node)
        if name is None or "_ARMY" not in name:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if (
                isinstance(key, ast.Attribute)
                and isinstance(key.value, ast.Name)
                and key.value.id == "UnitTypeId"
            ):
                roster.add(key.attr)
    return roster


# ===== TABLE SHAPE =====


def test_table_has_rows() -> None:
    """The table is the bot's scoring core — an empty one would silently zero every score."""
    assert len(SCORING_TABLE) > 0


def test_no_duplicate_pairs() -> None:
    """No pair may be listed twice — the second value would silently win."""
    pairs = _unit_type_pairs("SCORING_TABLE", TABLE_SOURCE)
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    assert not duplicates, f"duplicate SCORING_TABLE pairs: {duplicates}"


def test_keys_resolve_to_unit_type_ids() -> None:
    """Every key must be a pair of real UnitTypeId members.

    A typo'd key scores 0.0 through `.get(..., 0.0)` — invisible at runtime,
    fatal to the matchup the row was meant to express.
    """
    for our_type, enemy_type in _unit_type_pairs("SCORING_TABLE", TABLE_SOURCE):
        for member in (our_type, enemy_type):
            enum_name, _, member_name = member.partition(".")
            assert enum_name == "UnitTypeId", f"{member} is not a UnitTypeId member"
            assert member_name in UnitTypeId.__members__, f"{member} does not exist"


def test_runtime_keys_are_unit_type_id_pairs() -> None:
    """The imported table must hold (UnitTypeId, UnitTypeId) tuples."""
    for key in SCORING_TABLE:
        assert isinstance(key, tuple) and len(key) == 2, f"bad key shape: {key!r}"
        for member in key:
            assert isinstance(member, UnitTypeId), f"bad key member: {member!r}"


def test_values_on_one_scale() -> None:
    """All values are whole numbers on the table's single scale."""
    for key, value in SCORING_TABLE.items():
        assert isinstance(value, float), f"{key} has non-float value {value!r}"
        assert value.is_integer(), f"{key} has off-scale value {value}"
        assert SCORE_MIN <= value <= SCORE_MAX, f"{key} value {value} outside [{SCORE_MIN}, {SCORE_MAX}]"


def test_situational_rows_exist_in_table() -> None:
    """SITUATIONAL_ROWS only re-classifies existing rows — never invents one."""
    for pair in SITUATIONAL_ROWS:
        assert pair in SCORING_TABLE, f"stale situational row: {pair}"


def test_situational_rows_have_no_duplicates() -> None:
    """A duplicated set entry is silently dropped — keep the source honest."""
    pairs = _unit_type_pairs("SITUATIONAL_ROWS", TABLE_SOURCE)
    assert len(pairs) == len(set(pairs)), "duplicate SITUATIONAL_ROWS entries"


# ===== COVERAGE =====

def test_composition_units_have_attacker_rows() -> None:
    """Every unit the bot builds needs answers to the enemy types it can face."""
    roster = _composition_unit_types()
    assert roster, "no army compositions found in macro.py — parser drift?"
    attackers = {our_type for our_type, _ in SCORING_TABLE}
    missing = sorted(
        name for name in roster
        if UnitTypeId[name] not in attackers
    )
    assert not missing, f"composition units without attacker rows: {missing}"


def test_every_attacker_has_target_rows() -> None:
    """Every unit we score as an attacker must also be scoreable as an enemy.

    The mirror side is the PvP gap this table had: our own unit types appearing
    as the enemy (Dark Templar, Disruptor, High Templar, Archon, Sentry all had
    zero target rows before the coverage pass).
    """
    targets = {enemy_type for _, enemy_type in SCORING_TABLE}
    missing = sorted(
        our_type.name for our_type in {our for our, _ in SCORING_TABLE}
        if our_type not in targets
    )
    assert not missing, f"attackers without target rows: {missing}"


def test_excluded_units_have_no_attacker_rows() -> None:
    """Observer has no weapon; Oracle and Mothership are support — excluded by design."""
    excluded = {UnitTypeId.OBSERVER, UnitTypeId.ORACLE, UnitTypeId.MOTHERSHIP}
    attackers = {our_type for our_type, _ in SCORING_TABLE}
    assert not (attackers & excluded), f"excluded units used as attackers: {attackers & excluded}"
