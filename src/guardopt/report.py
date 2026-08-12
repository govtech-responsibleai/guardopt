"""An `OptimisationResult` as a self-contained Markdown report.

The "policy you can defend" pitch needs an artifact a reviewer can actually be handed:
one file with the three options side by side, every number's evidence, and the caveats
impossible to miss. This renders exactly what the result already contains — no new
prose is generated here, so the report cannot drift from the measurement.

**Limitations render first** on every recommendation, before the numbers they qualify.
A reader who stops early leaves knowing what the numbers cannot claim, not just the
numbers.
"""

from guardopt.domain.explain import format_percentage
from guardopt.optimise import OptimisationResult, ProfileRecommendation

__all__ = ["render_markdown"]


def _pct(value: float | None) -> str:
    formatted = format_percentage(value)
    return "—" if formatted is None else formatted


def _count(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _code_span(value: str) -> str:
    """Render a dataset value safely inside a Markdown inline code span.

    Case IDs come straight from a CSV with no charset restriction. A backtick in one would
    close the span and let whatever follows — `<img onerror=…>`, `[x](javascript:…)` — land
    as raw markup, which a renderer that passes HTML through (a wiki, pandoc, mdBook) would
    then execute. Stripping backticks (and flattening newlines, which also close a span)
    keeps the value inert text inside the span; the HTML report already escapes everything.
    """
    cleaned = value.replace("`", "").replace("\r", " ").replace("\n", " ")
    return f"`{cleaned}`"


def _ids(ids: tuple[str, ...]) -> str:
    return ", ".join(_code_span(case_id) for case_id in ids) if ids else "none"


def _summary_table(recommendations: tuple[ProfileRecommendation, ...]) -> list[str]:
    header = ["Metric"] + [r.profile.value.capitalize() for r in recommendations]
    rows = [
        ("Recall", lambda r: _pct(r.evaluated.recall)),
        ("Precision", lambda r: _pct(r.evaluated.precision)),
        ("F1", lambda r: _count(r.evaluated.f1)),
        ("False positives", lambda r: str(r.evaluated.false_positives)),
        ("False negatives", lambda r: str(r.evaluated.false_negatives)),
        (
            "Est. latency",
            lambda r: (
                "—"
                if r.evaluated.estimated_latency_ms is None
                else f"{round(r.evaluated.estimated_latency_ms)} ms"
            ),
        ),
        (
            "Est. cost / request",
            lambda r: (
                "—"
                if r.evaluated.estimated_cost is None
                else f"{r.evaluated.estimated_cost:.4g}"
            ),
        ),
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for label, cell in rows:
        lines.append(
            "| " + " | ".join([label] + [cell(r) for r in recommendations]) + " |"
        )
    return lines


def _recommendation_section(recommendation: ProfileRecommendation) -> list[str]:
    evaluated = recommendation.evaluated
    explanation = recommendation.explanation
    binary = evaluated.binary
    cm = evaluated.confusion_matrix

    lines: list[str] = [
        f"## {recommendation.profile.value.capitalize()}",
        "",
        explanation.headline,
        "",
        "### Limitations",
        "",
    ]
    lines.extend(f"- {limitation}" for limitation in explanation.limitations)

    lines.extend(["", "### What it measured", ""])
    lines.extend(f"{sentence}" for sentence in explanation.as_sentences())

    lines.extend(
        [
            "",
            "### Confusion matrix, with the cases behind every cell",
            "",
            f"- True positives ({cm.true_positives}): "
            + _ids(binary.true_positive_test_case_ids),
            f"- False positives ({cm.false_positives}): "
            + _ids(binary.false_positive_test_case_ids),
            f"- True negatives ({cm.true_negatives}): "
            + _ids(binary.true_negative_test_case_ids),
            f"- False negatives ({cm.false_negatives}): "
            + _ids(binary.false_negative_test_case_ids),
        ]
    )
    if binary.excluded_test_case_ids:
        lines.append(
            f"- Excluded ({len(binary.excluded_test_case_ids)}): "
            + _ids(binary.excluded_test_case_ids)
        )

    if recommendation.holdout is not None:
        holdout = recommendation.holdout
        lines.extend(
            [
                "",
                "### Holdout",
                "",
                f"On {holdout.case_count} cases held out of the search: recall "
                f"{_pct(holdout.recall)}, precision {_pct(holdout.precision)}.",
            ]
        )

    lines.extend(["", "### Guardrails", ""])
    if explanation.guardrails:
        lines.extend(f"- {line}" for line in explanation.guardrails)
    else:
        lines.append("- None. This policy enables no guardrails, so it blocks nothing.")

    if explanation.operations:
        lines.extend(["", explanation.operations])

    if recommendation.policy is not None:
        lines.extend(
            [
                "",
                "### Policy artifact",
                "",
                "```json",
                recommendation.policy.to_json(),
                "```",
            ]
        )
    return lines


def render_markdown(result: OptimisationResult, *, title: str = "guardopt report") -> str:
    """The whole result, one Markdown document, nothing invented."""
    diagnostics = result.diagnostics
    lines: list[str] = [
        f"# {title}",
        "",
        f"Search method: **{result.search_method.value}** — estimated space "
        f"{diagnostics.estimated_space_size}, evaluated "
        f"{diagnostics.evaluated_candidate_count}. "
        f"{result.pareto_candidate_count} distinct policies on the frontier.",
        "",
    ]

    if result.warnings:
        lines.extend(["## Notes and warnings", ""])
        lines.extend(f"- {warning}" for warning in result.warnings)
        lines.append("")

    if result.recommendations:
        lines.extend(["## The options, side by side", ""])
        lines.extend(_summary_table(result.recommendations))
        lines.append("")
        for recommendation in result.recommendations:
            lines.extend(_recommendation_section(recommendation))
            lines.append("")
    else:
        lines.append(
            "No recommendations could be made. The warnings above say why — that is an "
            "answer about this dataset, not a failure to produce one."
        )

    return "\n".join(lines).rstrip() + "\n"
