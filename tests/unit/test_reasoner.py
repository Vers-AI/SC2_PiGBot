"""Unit tests for the utility-AI reasoner library (bot/utilities/reasoner.py).

Purpose: pin the math every later adopter will build on — the response curve
    shapes, the canonical combination, the deterministic runner, and the
    IntelContext test seam.
Key Decisions: pure Python — no game, no SC2 binary, no manager import. The
    dependency-direction test parses the module's imports with `ast` and also
    loads the module standalone in a subprocess, because importing it through
    the `bot` package would run bot/__init__.py (which pulls in the game stack)
    and hide a library-side dependency.
Limitations: the tests pin the shapes and the arithmetic, not the personality
    values in config.yml — Task 8 refits those from telemetry.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from bot.utilities.reasoner import (
    Appraisal,
    BeliefSnapshot,
    Choice,
    Consideration,
    CurveParams,
    EconomySnapshot,
    IntelContext,
    apply_curve,
    combine,
    exponential,
    linear,
    load_curve_params,
    logistic,
    normalize,
    score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REASONER_SOURCE = REPO_ROOT / "bot" / "utilities" / "reasoner.py"
CONFIG_FILE = REPO_ROOT / "config.yml"

FORBIDDEN_IMPORT_ROOTS = ("bot", "sc2", "ares", "yaml", "numpy", "pandas")
"""Roots the library must never pull in — bot code (managers → library is the
dependency direction) and the game stack (the library has to stay pure)."""


def _params(
    kind: str = "linear",
    minimum: float = 0.0,
    maximum: float = 1.0,
    midpoint: float = 0.5,
    steepness: float = 4.0,
) -> CurveParams:
    """Curve parameters for a test — the library has no global curve state."""
    return CurveParams(
        kind=kind,
        minimum=minimum,
        maximum=maximum,
        midpoint=midpoint,
        steepness=steepness,
    )


# ===== RESPONSE CURVES =====


def test_linear_normalizes_the_raw_input() -> None:
    """A raw input inside the range lands proportionally on [0,1]."""
    assert linear(0.5, _params()) == pytest.approx(0.5)
    assert linear(5.0, _params(maximum=10.0)) == pytest.approx(0.5)
    assert linear(0.0, _params()) == pytest.approx(0.0)
    assert linear(1.0, _params()) == pytest.approx(1.0)


def test_linear_clamps_out_of_range_values() -> None:
    """Values outside the declared range clamp to the bounds — no extrapolation."""
    assert linear(-5.0, _params()) == pytest.approx(0.0)
    assert linear(7.0, _params()) == pytest.approx(1.0)


def test_linear_is_monotonic() -> None:
    """Monotonicity is what lets a curve act as an anti-oscillation tool."""
    values = [linear(x / 20, _params()) for x in range(21)]
    assert values == sorted(values)


def test_exponential_pins_its_endpoints() -> None:
    """The exponential curve starts at 0 and ends at 1, whatever the steepness."""
    for steepness in (-8.0, -1.0, 0.5, 4.0, 12.0):
        params = _params(kind="exponential", steepness=steepness)
        assert exponential(0.0, params) == pytest.approx(0.0, abs=1e-9)
        assert exponential(1.0, params) == pytest.approx(1.0, abs=1e-9)


def test_exponential_is_monotonic() -> None:
    """Both directions stay monotonic — a falling curve would invert a score."""
    for steepness in (-8.0, 4.0):
        params = _params(kind="exponential", steepness=steepness)
        values = [exponential(x / 20, params) for x in range(21)]
        assert values == sorted(values)


def test_exponential_convex_ignores_distant_noise() -> None:
    """Positive steepness is the threat-like shape: flat early, steep late."""
    params = _params(kind="exponential", steepness=4.0)
    assert exponential(0.25, params) < 0.1
    assert exponential(0.25, params) < exponential(0.5, params) < exponential(0.9, params)
    # The growth accelerates: the second half of the range moves more than the first.
    assert exponential(0.75, params) - exponential(0.5, params) > (
        exponential(0.5, params) - exponential(0.25, params)
    )


def test_exponential_concave_saturates() -> None:
    """Negative steepness is the value-like shape: fast rise, then saturation."""
    params = _params(kind="exponential", steepness=-4.0)
    assert exponential(0.25, params) > 0.5
    assert exponential(0.75, params) - exponential(0.5, params) < (
        exponential(0.5, params) - exponential(0.25, params)
    )


def test_exponential_zero_steepness_is_linear() -> None:
    """The zero-steepness limit is the identity — no shape, no surprise."""
    params = _params(kind="exponential", steepness=0.0)
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert exponential(x, params) == pytest.approx(x)


def test_logistic_reads_half_at_the_midpoint() -> None:
    """The midpoint is the tipping point: the curve reads exactly 0.5 there."""
    params = _params(kind="logistic", midpoint=0.3, steepness=8.0)
    assert logistic(0.3, params) == pytest.approx(0.5)
    assert logistic(0.7, _params(kind="logistic", midpoint=0.7)) == pytest.approx(0.5)
    # The midpoint moves with the parameter: below it the curve is under 0.5.
    assert logistic(0.3, _params(kind="logistic", midpoint=0.7)) < 0.5


def test_logistic_is_monotonic() -> None:
    """A falling sigmoid would turn a rising threat into a falling score."""
    params = _params(kind="logistic", midpoint=0.5, steepness=8.0)
    values = [logistic(x / 20, params) for x in range(21)]
    assert values == sorted(values)


def test_logistic_steepness_controls_commitment() -> None:
    """Steepness is the anti-oscillation knob: a sharper curve commits harder."""
    gentle = _params(kind="logistic", midpoint=0.5, steepness=2.0)
    sharp = _params(kind="logistic", midpoint=0.5, steepness=16.0)
    assert logistic(0.9, sharp) > logistic(0.9, gentle)
    assert logistic(0.1, sharp) < logistic(0.1, gentle)


def test_logistic_stays_inside_the_unit_range() -> None:
    """The shaped value is a probability-like scalar — never outside [0,1]."""
    params = _params(kind="logistic", midpoint=0.5, steepness=16.0)
    for x in (-3.0, 0.0, 0.5, 1.0, 3.0):
        assert 0.0 <= logistic(x, params) <= 1.0


def test_normalize_rejects_an_inverted_range() -> None:
    """A broken range would score garbage — it must fail loudly instead."""
    with pytest.raises(ValueError):
        normalize(0.5, 1.0, 1.0)


def test_apply_curve_dispatches_on_kind() -> None:
    """The config-driven entry point matches the direct shape calls."""
    for kind, shape in (
        ("linear", linear),
        ("exponential", exponential),
        ("logistic", logistic),
    ):
        params = _params(kind=kind)
        assert apply_curve(0.4, params) == pytest.approx(shape(0.4, params))


def test_apply_curve_rejects_an_unknown_kind() -> None:
    """An unknown shape is a configuration error, not a silent zero."""
    with pytest.raises(ValueError):
        apply_curve(0.4, _params(kind="sigmoid"))


# ===== CONFIG LOADING =====


def test_load_curve_params_reads_the_config_entry() -> None:
    """A named curve is read out of UtilityAI.Curves.<name>."""
    config = {
        "UtilityAI": {
            "Curves": {
                "threat": {
                    "kind": "exponential",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "steepness": 6.0,
                }
            }
        }
    }
    params = load_curve_params(config, "threat")
    assert params.kind == "exponential"
    assert params.maximum == pytest.approx(10.0)
    assert params.steepness == pytest.approx(6.0)


def test_load_curve_params_falls_back_to_the_shape_defaults() -> None:
    """A missing entry still yields a usable curve — the bot never crashes on config."""
    params = load_curve_params({}, "threat")
    assert params.kind == "linear"
    assert params.minimum == pytest.approx(0.0)
    assert params.maximum == pytest.approx(1.0)


def test_load_curve_params_ignores_non_numeric_values() -> None:
    """A malformed value falls back to the default instead of poisoning the score."""
    config = {"UtilityAI": {"Curves": {"threat": {"kind": "linear", "maximum": "oops"}}}}
    params = load_curve_params(config, "threat")
    assert params.maximum == pytest.approx(1.0)


def test_load_curve_params_rejects_an_unknown_kind() -> None:
    """An unknown shape is loud at load time, not wrong at score time."""
    config = {"UtilityAI": {"Curves": {"threat": {"kind": "sigmoid"}}}}
    with pytest.raises(ValueError):
        load_curve_params(config, "threat")


def test_load_curve_params_rejects_an_inverted_range() -> None:
    """An inverted range cannot normalize — it must fail loudly."""
    config = {"UtilityAI": {"Curves": {"threat": {"kind": "linear", "minimum": 5.0, "maximum": 1.0}}}}
    with pytest.raises(ValueError):
        load_curve_params(config, "threat")


def test_shipped_config_curves_all_load() -> None:
    """config.yml is the home of the curve parameters — every entry must load.

    Task 8 fits these values from telemetry; this test keeps the file and the
    loader in step (a renamed section would otherwise fail silently at runtime).
    """
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    curves = config["UtilityAI"]["Curves"]
    assert curves, "config.yml ships no response curves"
    for name in curves:
        params = load_curve_params(config, name)
        assert params.kind in ("linear", "exponential", "logistic")
        assert params.maximum > params.minimum


# ===== CANONICAL COMBINATION =====


def test_combine_sums_the_bases() -> None:
    """Bases are additive — value terms stack on top of each other."""
    combined = combine([Appraisal(base_score=2.0), Appraisal(base_score=3.0)])
    assert combined.base_score == pytest.approx(5.0)


def test_combine_multiplies_the_multipliers() -> None:
    """Multipliers are confidences — they scale the whole score."""
    combined = combine([Appraisal(multiplier=0.5), Appraisal(multiplier=0.8)])
    assert combined.multiplier == pytest.approx(0.4)


def test_combine_reports_a_veto() -> None:
    """A veto is surfaced in the fold, so the trail stays explainable."""
    combined = combine([Appraisal(base_score=5.0), Appraisal(veto=True)])
    assert combined.veto is True


def test_combine_of_nothing_scores_zero() -> None:
    """The empty sum is 0 — a candidate nobody scored must not win by default."""
    combined = combine([])
    assert combined.base_score == pytest.approx(0.0)
    assert combined.multiplier == pytest.approx(1.0)
    assert combined.veto is False


def test_canonical_score_is_bases_times_multipliers() -> None:
    """score = (Σ base_scores) × (Π multipliers) — the one currency."""
    choice = Choice(
        name="storm",
        appraisals=(
            Appraisal(base_score=2.0),
            Appraisal(base_score=1.0, multiplier=0.5),
        ),
    )
    ranked = score([choice])
    assert ranked[0].score == pytest.approx(1.5)


# ===== RUNNER =====


def test_score_ranks_candidates_by_descending_score() -> None:
    """The winner is the highest score — a plain argmax, no if-then gates."""
    ranked = score(
        [
            Choice(name="low", appraisals=(Appraisal(base_score=1.0),)),
            Choice(name="high", appraisals=(Appraisal(base_score=9.0),)),
            Choice(name="mid", appraisals=(Appraisal(base_score=5.0),)),
        ]
    )
    assert [entry.choice.name for entry in ranked] == ["high", "mid", "low"]
    assert [entry.rank for entry in ranked] == [0, 1, 2]


def test_score_breaks_ties_by_candidate_order() -> None:
    """Ties keep the input order (stable sort) — the argmax stays deterministic."""
    ranked = score(
        [
            Choice(name="first", appraisals=(Appraisal(base_score=4.0),)),
            Choice(name="second", appraisals=(Appraisal(base_score=4.0),)),
        ]
    )
    assert [entry.choice.name for entry in ranked] == ["first", "second"]
    assert ranked[0].score == ranked[1].score


def test_score_of_no_candidates_is_empty() -> None:
    """No candidates means no ranking — the caller decides what to do then."""
    assert score([]) == []


def test_score_zeroes_a_vetoed_candidate() -> None:
    """A hard impossibility outranks the arithmetic, even a negative one."""
    ranked = score(
        [
            Choice(name="impossible", appraisals=(Appraisal(base_score=-8.0, veto=True),)),
            Choice(name="possible", appraisals=(Appraisal(base_score=2.0),)),
        ]
    )
    scores = {entry.choice.name: entry.score for entry in ranked}
    assert scores["impossible"] == pytest.approx(0.0)
    assert scores["possible"] == pytest.approx(2.0)


def test_score_of_a_choice_without_considerations_is_zero() -> None:
    """An unscored candidate scores 0 — it never wins by accident."""
    ranked = score([Choice(name="empty")])
    assert ranked[0].score == pytest.approx(0.0)


def test_score_is_deterministic() -> None:
    """Same candidates in, same ranking out — the plan demands determinism."""
    choices = [
        Choice(name="a", appraisals=(Appraisal(base_score=0.3, multiplier=0.5),)),
        Choice(name="b", appraisals=(Appraisal(base_score=0.3, multiplier=0.5),)),
    ]
    first = [(entry.choice.name, entry.score) for entry in score(choices)]
    second = [(entry.choice.name, entry.score) for entry in score(choices)]
    assert first == second


# ===== INTEL CONTEXT =====


def test_intel_context_constructs_without_a_game() -> None:
    """The test seam: plain values, no bot instance, no SC2 binary."""
    context = IntelContext(
        belief=BeliefSnapshot(
            enemy_composition={"ROACH": 0.8, "ZERGLING": 0.4},
            strategy={"cheese": 0.98},
            confidence=0.75,
        ),
        threat_severity=0.6,
        intel_urgency=0.3,
        intel_freshness=0.9,
        economy=EconomySnapshot(
            worker_count=42, mineral_income=1800.0, gas_income=300.0, saturation=0.8
        ),
        tech_progress={"charge": 0.5},
    )
    assert context.belief.enemy_composition["ROACH"] == pytest.approx(0.8)
    assert context.belief.strategy["cheese"] == pytest.approx(0.98)
    assert context.threat_severity == pytest.approx(0.6)
    assert context.intel_urgency == pytest.approx(0.3)
    assert context.intel_freshness == pytest.approx(0.9)
    assert context.economy.worker_count == 42
    assert context.tech_progress["charge"] == pytest.approx(0.5)


def test_intel_context_is_frozen() -> None:
    """A decision cycle must not see the world change under it."""
    context = IntelContext()
    with pytest.raises(Exception):
        context.threat_severity = 0.9  # type: ignore[misc]


def test_intel_context_defaults_are_neutral() -> None:
    """The default context means 'nothing known, nothing threatened'."""
    context = IntelContext()
    assert context.threat_severity == pytest.approx(0.0)
    assert context.intel_freshness == pytest.approx(1.0)
    assert context.belief.confidence == pytest.approx(1.0)
    assert context.economy.worker_count == 0


# ===== CONSIDERATION PROTOCOL =====


class _ThreatConsideration:
    """Minimal consideration — the shape Tasks 4-6 implement against."""

    def __init__(self) -> None:
        self.steepness = 4.0

    def load(self, params) -> None:
        self.steepness = float(params.get("steepness", self.steepness))

    def evaluate(self, context: IntelContext) -> Appraisal:
        shaped = exponential(
            context.threat_severity,
            _params(kind="exponential", steepness=self.steepness),
        )
        return Appraisal(base_score=shaped)


def test_consideration_protocol_accepts_a_minimal_implementation() -> None:
    """The protocol is the stable seam — no inheritance required."""
    assert isinstance(_ThreatConsideration(), Consideration)


def test_considered_appraisals_feed_the_runner() -> None:
    """A consideration's appraisal is what the canonical combination consumes."""
    consideration = _ThreatConsideration()
    consideration.load({"steepness": 8.0})
    context = IntelContext(threat_severity=1.0)
    ranked = score(
        [
            Choice(name="respond", appraisals=(consideration.evaluate(context),)),
            Choice(name="ignore", appraisals=(Appraisal(base_score=0.1),)),
        ]
    )
    assert ranked[0].choice.name == "respond"
    assert ranked[0].score == pytest.approx(1.0)


# ===== DEPENDENCY DIRECTION =====


def test_reasoner_imports_no_bot_or_game_modules() -> None:
    """The library must not import manager code — the direction is managers → library."""
    tree = ast.parse(REASONER_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    roots = {name.split(".")[0] for name in imported}
    assert not (roots & set(FORBIDDEN_IMPORT_ROOTS)), (
        f"reasoner imports bot/game modules: {sorted(roots & set(FORBIDDEN_IMPORT_ROOTS))}"
    )


def test_reasoner_loads_standalone_without_bot_modules() -> None:
    """Loaded by path, the library pulls in nothing from the bot or the game.

    Importing it as `bot.utilities.reasoner` would run bot/__init__.py, which
    imports the whole game stack — that would mask a library-side dependency.
    """
    script = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('reasoner_standalone', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "leaked = sorted(n for n in sys.modules if n.split('.')[0] in "
        f"{set(FORBIDDEN_IMPORT_ROOTS)!r})\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(REASONER_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "LEAKED:" in result.stdout
    leaked = result.stdout.split("LEAKED:")[1].strip()
    assert leaked == "", f"reasoner pulled in {leaked}"
