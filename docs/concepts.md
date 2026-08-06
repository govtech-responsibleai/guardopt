# Concepts

Six ideas. Everything else in `guardopt` is built from them.

## Guardrail

A thing that reads text and returns a number. A toxicity classifier, a PII detector, a
prompt-injection model, a regex, your own validator. `guardopt` never runs one during
optimisation — it works from scores that already exist.

What it needs to know about each:

```python
from guardopt.domain.inputs import GuardrailDefinition
from guardopt.domain.types import ScoreDirection

GuardrailDefinition(
    name="toxicity",
    score_direction=ScoreDirection.HIGHER_IS_RISKIER,
    minimum_score=0.0,
    maximum_score=1.0,
)
```

### Score direction is stated, never guessed

`HIGHER_IS_RISKIER` or `LOWER_IS_RISKIER`. This is the single most important field on the
definition, and `guardopt` will not infer it under any circumstances.

The reason is that getting it wrong does not raise an error. It silently inverts every
threshold that guardrail contributes, so the policy blocks precisely the traffic it should
allow — and reports a confusion matrix, and metrics, and prose, all internally consistent
and all describing something you did not want. A wrong answer that looks like a right one
is worse than a crash.

Real detectors genuinely go both ways. A "refusal" score is high when the model *refused*,
which is usually the safe outcome. A "groundedness" score is high when the answer is well
supported. Neither is riskier-when-higher.

## Test case

One piece of labelled traffic, and what each guardrail scored on it.

```python
from guardopt.domain.inputs import GuardrailTestResult, TestCaseGuardrailResults
from guardopt.domain.types import ExpectedAction

TestCaseGuardrailResults(
    test_case_id="ticket-4471",
    expected_action=ExpectedAction.BLOCK,   # BLOCK is the positive class
    guardrail_results=[
        GuardrailTestResult(guardrail_name="toxicity", score=0.91),
        GuardrailTestResult(guardrail_name="pii", score=0.02),
        GuardrailTestResult(guardrail_name="injection", error="model unavailable"),
    ],
)
```

A result carries **either** a score **or** an error — never both, never neither. A row with
neither would be indistinguishable from an absent row, so it is rejected at validation.

### A gap is not a pass

Three different things can happen to a guardrail on a case, and they must not collapse:

| | Meaning |
|---|---|
| `score=0.02` | It ran, and found nothing |
| `error="..."` | It ran and failed, or could not run |
| no row at all | Nothing was ever recorded |

The last two are **not** clean passes. What they mean for the policy is your choice, set by
`OptimiserConfig.treat_missing_as`: either they behave as errors (which flag the request),
or the case is dropped from that policy's metrics and the count of dropped cases is
reported. What never happens is a missing score quietly counting in a guardrail's favour.

## Thresholds and the three bands

Each guardrail in a policy gets a **blocking** line and optionally a **flagging** line.
Together they cut the score range into three:

```text
HIGHER_IS_RISKIER:

  0.0 ──────────── warning ──────────── failed ──────────── 1.0
       PASS                 WARNING              FAIL
```

- **PASS** — through, untouched.
- **WARNING** — through, but recorded for review. Also called flagging.
- **FAIL** — blocked.

Both bands are **closed at their riskier edge**: `score >= failed` fails, `score >= warning`
warns. For `LOWER_IS_RISKIER` the whole picture reflects, and the comparisons become `<=`.

That inclusivity is not a detail. Candidate thresholds are generated *from observed scores*,
so cases land exactly on a threshold constantly — it is the common case, not an edge case.
Changing `>=` to `>` would silently reclassify them.

`warning=None` means the guardrail never flags: it blocks or it passes.

## Policy

A set of enabled guardrails, each with its thresholds. Turning a guardrail **off** is a
real move in the search — a guardrail that costs more in false positives than it earns in
recall should not be in your policy, and `guardopt` will drop it.

All enabled guardrails are evaluated, and the results combine:

- Any guardrail fails → the policy **fails**.
- Otherwise, any warning or error → the policy **warns**.
- Otherwise → the policy **passes**.

FAIL beats WARNING beats ERROR beats PASS. In particular a block outranks an error: a
policy that blocked the request blocked it, whatever else broke.

## Profile

`guardopt` does not return "the best policy", because which policy is best depends on what
a mistake costs you — and it does not know that.

Instead it finds the **Pareto frontier**: every policy where you cannot improve precision
without losing recall, or vice versa. Then it picks three points on it.

| Profile | Optimises for | Character |
|---|---|---|
| **Minimal** | F0.5 — precision-weighted | Fewest false positives. Misses more. |
| **Balanced** | F1 | The even trade. |
| **Strict** | F2 — recall-weighted | Catches the most. Costs false positives. |

If your data does not support three behaviourally distinct policies, you get fewer, plus a
warning saying why. Two policies that produce the identical verdict on every case are the
*same policy* however different their guardrail lists look — offering both as a choice
would be fabricating an option.

## Score matrix

The scored dataset, on its own:

```python
from guardopt.domain.matrix import ScoreMatrix

matrix = ScoreMatrix(guardrails=(...), cases=(...))
```

It is deliberately separate from the search settings, because they change for different
reasons — the matrix is data you measured, the config is how hard you want to look. It is
also the handover point between the two halves of the package:

```text
runtime.materialise(records, guards)  ->  ScoreMatrix    # calls guardrails, needs HTTP
optimise(matrix)                      ->  recommendations # pure, deterministic
```

If your scores are already in a spreadsheet, you skip the first line entirely — which is
the common case.
