"""The optimisation result as a single self-contained HTML file.

The Markdown report is for pull requests; this one is for the review meeting — openable
from a file share with no server, no external assets, no JavaScript. Same content rules
as the Markdown renderer: limitations lead every recommendation, unmeasured values
render as "—" never 0, and everything user-controlled is escaped (a guardrail named
`<script>` is someone's test fixture, not a foothold).

The one thing HTML can do that Markdown cannot is the trade-off chart: the whole Pareto
frontier as an inline SVG — every defensible policy, the three picks ringed — so the
"why this one" conversation happens over the actual shape of the trade-off rather than
three rows of a table. The chart draws only from measured values and is omitted, with a
note, when costs were never measured: an axis of invented zeros would be a picture of
nothing.
"""

import html
from guardopt.optimise import OptimisationResult, ProfileRecommendation

__all__ = ["render_html"]

_STYLE = """
  body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
         max-width: 60rem; padding: 0 1rem; color: #1d2a26; background: #fbfcfc;
         line-height: 1.55; }
  h1 { font-size: 1.6rem; } h2 { font-size: 1.2rem; margin-top: 2.2rem; }
  table { border-collapse: collapse; margin: 1rem 0; }
  th, td { border-bottom: 1px solid #d7dedb; padding: .35rem .8rem; text-align: left;
           font-size: .92rem; }
  th { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em;
       color: #5d6b66; }
  .warn { background: #f7edd4; border-left: 4px solid #a97b12; padding: .6rem 1rem;
          margin: .8rem 0; }
  .limits { background: #edf3f0; border-left: 4px solid #17685a; padding: .6rem 1rem;
            margin: .8rem 0; }
  .meta { color: #75837e; font-size: .85rem; }
  details { margin: .6rem 0; } summary { cursor: pointer; color: #17685a; }
  pre { background: #eef1f0; padding: .8rem 1rem; overflow-x: auto; font-size: .82rem; }
  svg text { font-family: inherit; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def _summary_table(recommendations: tuple[ProfileRecommendation, ...]) -> str:
    heads = "".join(
        f"<th>{_esc(r.profile.value.capitalize())}</th>" for r in recommendations
    )
    rows = [
        ("Recall", lambda r: _pct(r.evaluated.recall)),
        ("Precision", lambda r: _pct(r.evaluated.precision)),
        ("F1", lambda r: _num(r.evaluated.f1)),
        ("False positives", lambda r: str(r.evaluated.false_positives)),
        ("False negatives", lambda r: str(r.evaluated.false_negatives)),
        (
            "Est. latency",
            lambda r: "—"
            if r.evaluated.estimated_latency_ms is None
            else f"{round(r.evaluated.estimated_latency_ms)} ms",
        ),
        ("Est. cost / request", lambda r: _num(r.evaluated.estimated_cost, 4)),
    ]
    body = "".join(
        "<tr><td>"
        + _esc(label)
        + "</td>"
        + "".join(f"<td>{_esc(cell(r))}</td>" for r in recommendations)
        + "</tr>"
        for label, cell in rows
    )
    return f"<table><tr><th>Metric</th>{heads}</tr>{body}</table>"


def _frontier_chart(result: OptimisationResult) -> str:
    """Cost-vs-F1 scatter of the whole frontier, the profile picks ringed.

    Cost effectiveness would need a division by zero-able costs; raw cost with "left is
    cheaper" labelling keeps every measured point plottable.
    """
    points: list[tuple[float, float, int]] = []  # (cost, f1, id of the evaluated)
    for policy in result.frontier:
        if policy.estimated_cost is not None and policy.f1 is not None:
            points.append((policy.estimated_cost, policy.f1, id(policy)))
    if len(points) < 2:
        return (
            "<p class='meta'>No trade-off chart: fewer than two frontier policies "
            "carry both a measured cost and a measurable F1.</p>"
        )

    picked = {id(r.evaluated) for r in result.recommendations}
    min_c, max_c = min(c for c, _, _ in points), max(c for c, _, _ in points)
    min_f, max_f = min(f for _, f, _ in points), max(f for _, f, _ in points)
    span_c = (max_c - min_c) or 1.0
    span_f = (max_f - min_f) or 1.0

    width, height, pad = 640, 320, 45

    def x_of(cost: float) -> float:
        return pad + (cost - min_c) / span_c * (width - 2 * pad)

    def y_of(f1: float) -> float:
        return height - pad - (f1 - min_f) / span_f * (height - 2 * pad)

    shapes = [
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        f'stroke="#c6cecb"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{pad}" y2="{pad}" stroke="#c6cecb"/>',
        f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-size="12" '
        f'fill="#5d6b66">cost per request (left = cheaper)</text>',
        f'<text x="14" y="{height / 2}" text-anchor="middle" font-size="12" '
        f'fill="#5d6b66" transform="rotate(-90 14 {height / 2})">F1</text>',
    ]
    for cost, f1, identity in points:
        x, y = x_of(cost), y_of(f1)
        shapes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#17685a"/>')
        if identity in picked:
            shapes.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="none" '
                f'stroke="#1d2a26" stroke-width="1.5"/>'
            )
    for recommendation in result.recommendations:
        evaluated = recommendation.evaluated
        if evaluated.estimated_cost is None or evaluated.f1 is None:
            continue
        x, y = x_of(evaluated.estimated_cost), y_of(evaluated.f1)
        shapes.append(
            f'<text x="{x + 13:.1f}" y="{y - 8:.1f}" font-size="12" fill="#1d2a26">'
            f"{_esc(recommendation.profile.value.capitalize())}</text>"
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Pareto frontier: cost per request against F1">'
        + "".join(shapes)
        + "</svg>"
        + f"<p class='meta'>{len(points)} frontier policies with measured cost; "
        f"ringed points are the recommended profiles.</p>"
    )


def _recommendation_section(recommendation: ProfileRecommendation) -> str:
    explanation = recommendation.explanation
    parts = [f"<h2>{_esc(explanation.profile.value.capitalize())}</h2>"]
    parts.append(f"<p>{_esc(explanation.headline)}</p>")

    # Limitations FIRST — same order of honesty as the Markdown report.
    if explanation.limitations:
        items = "".join(f"<li>{_esc(line)}</li>" for line in explanation.limitations)
        parts.append(f"<div class='limits'><strong>Read first</strong><ul>{items}</ul></div>")

    parts.append(f"<p>{_esc(explanation.coverage)} {_esc(explanation.accuracy)}</p>")
    if explanation.flagging is not None:
        parts.append(f"<p>{_esc(explanation.flagging)}</p>")

    holdout = recommendation.holdout
    if holdout is not None:
        line = (
            f"Holdout ({holdout.case_count} cases): recall {_pct(holdout.recall)}, "
            f"precision {_pct(holdout.precision)}."
        )
        if holdout.false_negative_bound is not None:
            line += " " + holdout.false_negative_bound.sentence()
        parts.append(f"<p>{_esc(line)}</p>")

    items = "".join(f"<li>{_esc(line)}</li>" for line in explanation.guardrails)
    parts.append(f"<ul>{items}</ul>")

    if recommendation.policy is not None:
        parts.append(
            "<details><summary>Policy artifact (JSON)</summary><pre>"
            + _esc(recommendation.policy.to_json())
            + "</pre></details>"
        )
    return "".join(parts)


def render_html(result: OptimisationResult, *, title: str = "guardopt report") -> str:
    """The whole result as one HTML document, self-contained and escaped throughout."""
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{_esc(title)}</title><style>{_STYLE}</style></head><body>",
        f"<h1>{_esc(title)}</h1>",
        f"<p class='meta'>Search: {_esc(result.search_method.value)} · "
        f"{result.diagnostics.evaluated_candidate_count} candidates evaluated · "
        f"{result.pareto_candidate_count} on the frontier</p>",
    ]
    for warning in result.warnings:
        parts.append(f"<div class='warn'>{_esc(warning)}</div>")
    if result.stability is not None:
        parts.append(f"<p>{_esc(result.stability.sentence())}</p>")

    if result.recommendations:
        parts.append(_summary_table(result.recommendations))
        parts.append("<h2>The trade-off</h2>")
        parts.append(_frontier_chart(result))
        for recommendation in result.recommendations:
            parts.append(_recommendation_section(recommendation))
    else:
        parts.append("<p>No recommendations could be made.</p>")

    parts.append("</body></html>")
    return "".join(parts)
