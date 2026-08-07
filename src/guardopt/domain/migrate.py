"""Reading a v1 route policy into the v2 model.

v1 files exist, so they have to be readable. But v1 records less than v2 needs, and the gap
is not cosmetic: **v1 carries no score direction.** It assumed higher is riskier, globally
and silently.

Loading a v1 file under that assumption without saying so is the failure this package
refuses everywhere else. A guardrail whose *low* scores are the risky ones would come out
with every threshold inverted — blocking precisely the traffic it should allow, and
reporting a confusion matrix, metrics and prose that are all internally consistent and all
describe something nobody wanted.

So conversion is not automatic. The caller states the direction per guardrail, or
explicitly accepts v1's blanket assumption. There is no third option and no default.

The other loss is narrower and also refused rather than papered over: v1 could set
thresholds per (guard, label), and a v2 binding is per guardrail. Those overrides have
nowhere to go, so a file carrying them stops rather than silently dropping a number
somebody tuned.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from guardopt.domain.policy import GuardrailBinding, Policy, Stage
from guardopt.domain.types import ScoreDirection, StageCondition

__all__ = ["V1_SCHEMA_VERSION", "load_v1"]

V1_SCHEMA_VERSION = "guardrail-router.policy.v1"


def _resolve_directions(
    guard_names: Sequence[str],
    score_directions: Mapping[str, ScoreDirection] | None,
    assume_higher_is_riskier: bool,
) -> dict[str, ScoreDirection]:
    if score_directions is not None and assume_higher_is_riskier:
        raise ValueError(
            "pass score_directions or assume_higher_is_riskier, not both — two sources of "
            "truth for the same fact is how they come to disagree"
        )

    if score_directions is None and not assume_higher_is_riskier:
        raise ValueError(
            "a v1 policy records no score direction, and this package never infers one: a "
            "guessed direction inverts every threshold that guardrail contributes. Pass "
            "score_directions={name: ScoreDirection...} to state them, or "
            "assume_higher_is_riskier=True to accept v1's own blanket assumption."
        )

    if assume_higher_is_riskier:
        return {name: ScoreDirection.HIGHER_IS_RISKIER for name in guard_names}

    assert score_directions is not None  # narrowing; the branches above cover the rest
    missing = [name for name in guard_names if name not in score_directions]
    if missing:
        raise ValueError(
            f"no score direction given for {sorted(missing)}. Defaulting the rest would "
            f"reintroduce exactly the assumption this argument exists to remove."
        )
    return dict(score_directions)


def _thresholds_for(
    guard_name: str, overrides: Mapping[str, Any], low: float, high: float
) -> tuple[float, float]:
    """The (low, high) pair this guardrail runs at, after v1's override resolution.

    Only guard-scoped overrides survive the move to v2; label-scoped ones are rejected by
    the caller before this runs.
    """
    guard_override = dict(overrides.get("guards") or {}).get(guard_name)
    if guard_override:
        return float(guard_override["low"]), float(guard_override["high"])
    return low, high


def load_v1(
    payload: Mapping[str, Any],
    *,
    score_directions: Mapping[str, ScoreDirection] | None = None,
    assume_higher_is_riskier: bool = False,
) -> Policy:
    """Convert a v1 route policy into a v2 `Policy`.

    Exactly one of `score_directions` or `assume_higher_is_riskier` must be given.

    Raises rather than converting when the result would be wrong or lossy: an unreadable
    version, a `low >= high` pair, a label-scoped threshold override, or a guardrail
    declared `LOWER_IS_RISKIER` — see below for why that last one is impossible.
    """
    version = str(payload.get("schema_version", ""))
    if version != V1_SCHEMA_VERSION:
        raise ValueError(
            f"load_v1 reads schema_version {V1_SCHEMA_VERSION!r}, got {version!r}"
        )

    low = float(payload.get("low_threshold", 0.2))
    high = float(payload.get("high_threshold", 0.8))
    if low >= high:
        raise ValueError(
            f"v1 requires low_threshold < high_threshold, got {low} and {high}; a file "
            f"breaking its own invariant was not written by the v1 writer, and guessing "
            f"which way round the author meant them is not this loader's business"
        )

    overrides = dict(payload.get("thresholds") or {})
    # An empty skeleton is not an override anyone tuned, so only non-empty ones stop us.
    if overrides.get("labels") or overrides.get("guard_labels"):
        raise ValueError(
            "this v1 policy sets thresholds per label, and a v2 binding is per guardrail, "
            "so there is nowhere to put them. Dropping a threshold somebody tuned would "
            "change what the policy blocks without saying so. Fan the guardrail out into "
            "per-label signals first (see guardopt.domain.fanout), then convert."
        )

    stages_payload = list(payload["stages"])
    guard_names = [
        str(name) for stage in stages_payload for name in stage.get("guards", [])
    ]
    directions = _resolve_directions(
        guard_names, score_directions, assume_higher_is_riskier
    )

    stages: list[Stage] = []
    for stage_payload in stages_payload:
        bindings: list[GuardrailBinding] = []
        for raw_name in stage_payload.get("guards", []):
            name = str(raw_name)
            direction = directions[name]

            # v1's pair is ordered low < high, which for a lower-is-riskier guardrail says
            # the opposite of what it means. There is no honest conversion: the numbers
            # themselves are wrong, not merely their labels. Emitting them would produce a
            # policy that blocks the safe traffic.
            if direction is ScoreDirection.LOWER_IS_RISKIER:
                raise ValueError(
                    f"'{name}' is declared {direction.value}, but v1 stores thresholds as "
                    f"low < high and assumed the opposite. The recorded numbers cannot be "
                    f"reinterpreted for this guardrail — re-derive its thresholds from "
                    f"scored data rather than converting them."
                )

            warning, failed = _thresholds_for(name, overrides, low, high)
            bindings.append(
                GuardrailBinding(
                    name=name,
                    score_direction=direction,
                    failed=failed,
                    warning=warning,
                )
            )

        stages.append(
            Stage(
                name=str(stage_payload["name"]),
                guardrails=tuple(bindings),
                parallel=bool(stage_payload.get("parallel", True)),
                condition=StageCondition(stage_payload.get("condition", "always")),
                allow_exit=bool(stage_payload.get("allow_exit", False)),
                resolves_uncertainty=bool(
                    stage_payload.get("resolves_uncertainty", False)
                ),
            )
        )

    return Policy(name=str(payload["name"]), stages=tuple(stages))
