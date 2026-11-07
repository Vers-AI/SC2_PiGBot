"""Utility-AI reasoner — the shared scoring library every decision system uses.

Purpose: One currency for the bot's decisions — response curves that normalize
         raw inputs, the appraisal a consideration returns, the canonical
         combination runner, and the IntelContext snapshot that considerations
         score against. Tasks 4-6 (macro, reactions, scouting) and the Phase 2
         adopters all score through this module instead of growing thresholds.
Key Decisions: Canonical combination is score = (Σ base_scores) × (Π multipliers),
               any veto ⇒ 0. Every raw input passes a response curve first
               (normalize to [0,1], then shape) — no raw values enter a score.
               Curve parameters are personality hypotheses loaded from config.yml
               (UtilityAI.Curves.<name>). The module imports nothing from the
               bot, so the dependency direction stays managers → library.
Limitations: The library scores; it decides nothing. Curve parameters are seeded
              values — Task 8 refits them from telemetry. Importing the module
              through the `bot` package still runs bot/__init__.py (which pulls
              in the game stack); the library itself has no such dependency.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

CONFIG_SECTION = "UtilityAI"
"""config.yml section holding the response-curve personality hypotheses."""

CURVE_KINDS = ("linear", "exponential", "logistic")
"""Response-curve shapes the library implements."""

_PARAM_FIELDS = ("minimum", "maximum", "midpoint", "steepness")


@dataclass(frozen=True)
class CurveParams:
    """Parameters of one named response curve.

    kind: which shape to apply — "linear", "exponential" or "logistic".
    minimum/maximum: the raw input's range; the curve normalizes onto [0,1].
    midpoint: the raw input value the logistic curve reads as 0.5.
    steepness: shape knob. Exponential — sign picks the direction (positive:
               convex commitment shape, negative: concave saturating shape);
               logistic — slope at the midpoint.

    Bounds belong to the *quantity* being scored (a worker count runs 0-70, a
    severity 0-1), which is why each curve is configured by name instead of by
    shape: one shape, many quantities, each with its own range.
    """

    kind: str
    minimum: float
    maximum: float
    midpoint: float
    steepness: float


DEFAULT_CURVE_PARAMS: Mapping[str, CurveParams] = MappingProxyType(
    {
        "linear": CurveParams(
            kind="linear", minimum=0.0, maximum=1.0, midpoint=0.5, steepness=4.0
        ),
        "exponential": CurveParams(
            kind="exponential", minimum=0.0, maximum=1.0, midpoint=0.5, steepness=4.0
        ),
        "logistic": CurveParams(
            kind="logistic", minimum=0.0, maximum=1.0, midpoint=0.5, steepness=8.0
        ),
    }
)
"""Per-shape defaults — the fallback for every field a config entry omits."""


@dataclass(frozen=True)
class Appraisal:
    """One consideration's verdict on one candidate.

    base_score: the additive term. Costs are negative bases, never vetoes.
    multiplier: the multiplicative confidence — beliefs and staleness enter
                here, so stale intel discounts a score instead of zeroing it.
    veto: hard impossibility (no production facility, no worker to send). Any
           veto zeroes the candidate's score outright.
    """

    base_score: float = 0.0
    multiplier: float = 1.0
    veto: bool = False


@dataclass(frozen=True)
class Choice:
    """One candidate action: a stable name plus the appraisals that score it.

    The name is how the caller maps the winner back to its action (a
    composition, a reaction handler, a scout mission) — the library stays free
    of game concepts.
    """

    name: str
    appraisals: tuple[Appraisal, ...] = ()


@dataclass(frozen=True)
class RankedChoice:
    """A candidate with its canonical score and rank (0 = the winner)."""

    choice: Choice
    score: float
    rank: int


@dataclass(frozen=True)
class BeliefSnapshot:
    """Belief state as the reasoner sees it — probabilities, never game objects.

    enemy_composition: enemy unit type name → P(unit present), the
                        probability-weighted army belief.
    strategy: strategy category → P(category), e.g. {"cheese": 0.98}.
    confidence: how much the belief layer trusts the picture as a whole; it
                enters scores as a multiplier.
    """

    enemy_composition: Mapping[str, float] = field(default_factory=dict)
    strategy: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass(frozen=True)
class EconomySnapshot:
    """Our economy at the re-score cadence.

    worker_count: workers alive. mineral_income/gas_income: collection rates
                  per minute. saturation: how much of the mineral base is
                  worked out (0-1).
    """

    worker_count: int = 0
    mineral_income: float = 0.0
    gas_income: float = 0.0
    saturation: float = 0.0


@dataclass(frozen=True)
class IntelContext:
    """The snapshot every consideration scores against — the reasoner's test seam.

    Built by the intel layer on the re-score cadence, never per frame. Frozen:
    a decision cycle must not see the world change under it. Every field is a
    normalized scalar or probability, so no raw game state reaches the scoring
    math — raw inputs go through a response curve first.

    Unit and scenario tests construct this directly with plain values; no game,
    no bot instance, no SC2 binary is involved.
    """

    belief: BeliefSnapshot = field(default_factory=BeliefSnapshot)
    threat_severity: float = 0.0
    intel_urgency: float = 0.0
    intel_freshness: float = 1.0
    economy: EconomySnapshot = field(default_factory=EconomySnapshot)
    tech_progress: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class Consideration(Protocol):
    """The shape every consideration implements — the stable seam for Tasks 4-6.

    load(params): read the tunable parameters this consideration needs out of
                  the config-derived mapping. Called once, before scoring.
    evaluate(context): score the current situation; returns an Appraisal whose
                       base/multiplier/veto feed the canonical combination.
    """

    def load(self, params: Mapping[str, Any]) -> None:
        ...

    def evaluate(self, context: IntelContext) -> Appraisal:
        ...


# ===== RESPONSE CURVES =====


def normalize(value: float, minimum: float, maximum: float) -> float:
    """Map a raw input onto [0,1]; out-of-range values clamp to the bounds.

    Raises:
        ValueError: when the range is inverted — a broken curve would silently
            score garbage, so the misconfiguration must be loud.
    """
    if maximum <= minimum:
        raise ValueError(
            f"curve range must satisfy maximum > minimum, got [{minimum}, {maximum}]"
        )
    return min(1.0, max(0.0, (value - minimum) / (maximum - minimum)))


def linear(value: float, params: CurveParams) -> float:
    """Normalized identity — the baseline curve.

    Use it when a raw input already means what the score should mean (a
    probability, a counter-table value scaled onto the score range).
    """
    return normalize(value, params.minimum, params.maximum)


def exponential(value: float, params: CurveParams) -> float:
    """Exponential shaping of the normalized input.

    steepness > 0 → convex: distant noise is ignored and commitment arrives
    near the top of the range — the threat-like shape the plan asks for.
    steepness < 0 → concave: fast rise, then saturation (value-like shapes).
    steepness = 0 → the linear limit.
    """
    x = normalize(value, params.minimum, params.maximum)
    steepness = params.steepness
    if abs(steepness) < 1e-6:
        return x
    # Written with only non-positive exponents so a large steepness cannot
    # overflow: (e^(k*x) - 1) / (e^k - 1), scaled to keep both terms in range.
    if steepness > 0:
        shaped = (
            math.exp(steepness * (x - 1.0)) - math.exp(-steepness)
        ) / (1.0 - math.exp(-steepness))
    else:
        magnitude = -steepness
        shaped = (1.0 - math.exp(-magnitude * x)) / (1.0 - math.exp(-magnitude))
    return min(1.0, max(0.0, shaped))


def logistic(value: float, params: CurveParams) -> float:
    """Sigmoid shaping around a midpoint — the default for response decisions.

    The curve reads 0.5 exactly at the midpoint and rises monotonically; a
    larger steepness makes the bot commit harder around the tipping point.
    """
    x = normalize(value, params.minimum, params.maximum)
    midpoint = normalize(params.midpoint, params.minimum, params.maximum)
    z = params.steepness * (x - midpoint)
    # Two branches so only non-positive values reach exp() — no overflow.
    if z >= 0:
        shaped = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        shaped = exp_z / (1.0 + exp_z)
    return min(1.0, max(0.0, shaped))


def apply_curve(value: float, params: CurveParams) -> float:
    """Shape a raw input with the curve `params` describes (dispatch on kind)."""
    if params.kind == "exponential":
        return exponential(value, params)
    if params.kind == "logistic":
        return logistic(value, params)
    if params.kind == "linear":
        return linear(value, params)
    raise ValueError(f"unknown curve kind: {params.kind!r}")


# ===== CANONICAL COMBINATION =====


def combine(appraisals: Iterable[Appraisal]) -> Appraisal:
    """Fold one box's appraisals: sum of bases, product of multipliers.

    A veto is reported rather than applied here — the caller zeroes the score,
    so a vetoed box stays visible in the appraisal trail.
    """
    base_score = 0.0
    multiplier = 1.0
    veto = False
    for appraisal in appraisals:
        base_score += appraisal.base_score
        multiplier *= appraisal.multiplier
        veto = veto or appraisal.veto
    return Appraisal(base_score=base_score, multiplier=multiplier, veto=veto)


def score(choices: Iterable[Choice]) -> list[RankedChoice]:
    """Rank candidates by the canonical combination — deterministic argmax.

    score(choice) = (Σ base_scores) × (Π multipliers), any veto ⇒ 0.

    The winner is ranked[0]. Ties keep the candidate's input order (stable
    sort), so the argmax is reproducible run to run.
    """
    scored = [(choice, _score_choice(choice)) for choice in choices]
    ordered = sorted(scored, key=lambda pair: -pair[1])
    return [
        RankedChoice(choice=choice, score=value, rank=rank)
        for rank, (choice, value) in enumerate(ordered)
    ]


def _score_choice(choice: Choice) -> float:
    """Canonical score of one candidate; a veto wins over the arithmetic."""
    combined = combine(choice.appraisals)
    if combined.veto:
        return 0.0
    return combined.base_score * combined.multiplier


# ===== CONFIG LOADING =====


def load_curve_params(config: Mapping[str, Any], name: str) -> CurveParams:
    """Read one named curve out of the bot's config mapping.

    `config` is the parsed config.yml (bot.config) and the curve lives under
    UtilityAI.Curves.<name>:

        UtilityAI:
            Curves:
                threat:
                    kind: exponential
                    minimum: 0.0
                    maximum: 1.0
                    steepness: 4.0

    Missing fields fall back to the shape's defaults (DEFAULT_CURVE_PARAMS);
    values that are not numbers are ignored. An unknown kind or an inverted
    range raises ValueError: both are configuration mistakes that must be loud
    at load time instead of silently wrong scores in the game loop.
    """
    entry = _mapping(_lookup(config, CONFIG_SECTION, "Curves", name))
    kind = str(entry.get("kind", "linear"))
    if kind not in DEFAULT_CURVE_PARAMS:
        raise ValueError(f"unknown curve kind {kind!r} for curve {name!r}")
    overrides: dict[str, float] = {}
    for field_name in _PARAM_FIELDS:
        if field_name not in entry:
            continue
        try:
            overrides[field_name] = float(entry[field_name])
        except (TypeError, ValueError):
            continue
    params = replace(DEFAULT_CURVE_PARAMS[kind], **overrides)
    if params.maximum <= params.minimum:
        raise ValueError(
            f"curve {name!r} has an inverted range [{params.minimum}, {params.maximum}]"
        )
    return params


def _mapping(value: Any) -> Mapping[str, Any]:
    """Empty mapping for anything that is not a mapping (missing config key)."""
    return value if isinstance(value, Mapping) else {}


def _lookup(config: Mapping[str, Any], *keys: str) -> Any:
    """Walk a nested mapping, tolerating a missing or malformed section."""
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current
