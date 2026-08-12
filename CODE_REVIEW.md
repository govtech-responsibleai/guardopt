# Codebase Review

*guardopt `0.2.0.dev0` — production-readiness and long-term maintainability audit.*

This review covers the `src/guardopt` package (60 modules, ~12.6k LOC). It was produced by
reading the source directly, running the project's own tooling, tracing execution across
modules, and cross-checking four independent deep-dives (architecture, security,
performance, tests) against the code. Every substantive claim below was verified against
the implementation; measured numbers were reproduced in a throwaway harness on the repo's
own venv. Findings the reviews raised that did not survive verification are recorded in
§8 so the effort spent on them is visible.

**Headline:** this is an unusually disciplined codebase. The offline core is pure and
deterministic, the pure/vectorised parity is genuinely well-guarded by property tests, the
"never guess, refuse instead" posture is real and consistently applied, and credential/TLS
hygiene on the network edges holds up. The issues worth acting on are concentrated on the
**live-traffic and output-generation edges** (a shipped ReDoS regex, a code-generator that
interpolates unescaped names), one **systemic type-safety gap** on the recommendation path,
and a handful of **provably-equivalent performance wins** in the search hot loop.

---

## 0. Resolution status

> **Update — remediation applied.** Every finding below except **F16 (add CI)** has been
> fixed in the working tree. The gate is green: `ruff` clean, `mypy` clean (62 source
> files), **869 tests pass** (the original 860 plus 9 new regression tests). The findings
> in §4 are left as the original audit record; each now carries a **Status** line, and this
> table is the index. F16 was deliberately excluded at the maintainer's direction (adding CI
> is a repo/infra decision, not a code change); its symptom — the stale test badge — is left
> untouched because it belongs to that finding.

| ID | Finding | Severity | Status |
| --- | --- | --- | --- |
| F1 | ReDoS in shipped email PII regex | High | ✅ Fixed — linear bounded/possessive regex + `max_scan_chars` cap; regression test T6 |
| F2 | RCE via code-gen exporters | High | ✅ Fixed — `safe_for_generated_source()` on names in both exporters; injection regression test |
| F3 | Central type erased to `Any` + `getattr` | High | ✅ Fixed — read-only `RankablePolicy` Protocol; all 28 `getattr(…, default)` removed |
| F4 | Evaluation semantics implemented 3× | Medium | ✅ Fixed — shared `apply_stage_transition` drives both the offline walk and the runtime router |
| F5 | Process-global non-reclaimable thread pool | Medium | ✅ Fixed — `executor=` injectable per `GuardrailRouter`/`materialise` for isolation |
| F6 | Result types buried in 905-line `search.py` | Medium | ✅ Fixed — moved to `domain/evaluation.py` (+ shared `assemble_evaluated_policy`) |
| F7 | `retune` re-invokes private `_split_for_holdout` | Medium | ✅ Fixed — `split_for_holdout` now public with a stated determinism contract |
| F8 | No package exception base | Medium | ✅ Fixed — `GuardoptError` / `GuardoptInputError` / `SearchSpaceError` hierarchy |
| F9 | Threshold-invariant latency/cost recomputed | High (perf) | ✅ Fixed — memoised on `enabled_names` |
| F10 | Per-case ID lists built for every candidate | Medium | ✅ Fixed — `with_ids=False` in search; public `populate_case_ids` rebuilds for the picks |
| F11 | Cache memory O(space × n_cases) | Medium | ✅ Fixed — `outcome_signature` is now compact `bytes` |
| F12 | Per-guardrail masks recomputed per candidate | Medium | ✅ Fixed — mask cache keyed on `(name, thresholds)` |
| F13 | `on_decision` uncontained on the request path | Low–Med | ✅ Fixed — contained (logged + swallowed); regression test T3 |
| F14 | Markdown report performs no output escaping | Low | ✅ Fixed — case IDs neutralised for the code span; regression test |
| F15 | Staged search recomputes invariants in the loop | Low | ✅ Fixed — `guardrail_by_name` bound once; per-stage latency memoised |
| **F16** | **No CI; stale gate** | **Med–High** | **⛔ Not fixed — excluded by request** (infra decision; symptom badge left as-is) |
| F17 | No lockfile / no dep upper bounds | Medium | ✅ Partially — conservative `<3` caps on numpy/pydantic (a lockfile needs network, unavailable here) |
| F18 | Profile→objective mapping duplicated 3× | Low | ✅ Fixed — single `metrics.PROFILE_BETA` source |

**Measured impact of the performance fixes (F9–F12, F15)**, on the review's own 500-case /
5-guardrail / 228,096-policy exhaustive workload: **27.6 s → 9.3 s (~3×)** wall and
**2.64 GB → 605 MB (~4.4×)** peak RSS, with results provably unchanged (the pure↔vectorised
parity tests still hold, now swept at wider shapes by T5).

Two new source modules — `domain/errors.py` (F8) and `domain/evaluation.py` (F6) — and one
new test module, `tests/test_review_regressions.py` (T1–T6 plus the F2/F14 guards). Eight
existing test files absorbed internal-representation changes (the `bytes` signature and the
`RankablePolicy` shape) — behaviour-preserving stub updates, not test-weakening.

---

## 1. Executive assessment

| Area | Assessment | Key concern |
| --- | --- | --- |
| Maintainability | **Strong**, one systemic gap | The central result type is erased to `Any` and read with `getattr(…, default)` across the entire recommendation-deciding path — mypy checks none of it. |
| Architecture / code quality | **Strong** | The evaluation semantics exist in three hand-maintained implementations (pure, NumPy, runtime router); parity is enforced by tests, not structure. `search.py` is a 905-line god-module holding the most-consumed domain types. |
| Performance | **Good, with clear wins left** | Threshold-invariant work (latency, cost, per-guardrail masks, ID lists) is recomputed for every one of ~10⁵ candidates; cache memory is O(space × n_cases) → 2.6 GB on a 500-case exhaustive run. |
| Security | **Good posture, sharp edges** | Quadratic ReDoS in the shipped email PII regex, reachable from live request text; unescaped `policy.name` → RCE in the code-gen exporters; a process-global, non-reclaimable sync-guardrail thread pool that the ReDoS weaponises. |
| Tests / engineering | **Excellent suite, weak automation** | No CI (the README badge is already stale: 852 vs 860 actual); a documented concurrency invariant (`reload_policy` under in-flight requests) is untested; no dependency lockfile on a NumPy hot path. |

### The 3–5 most important findings

1. **[F1 · High] ReDoS in the shipped `pii_guardrail()` email pattern**, reachable from
   attacker-controlled request text. Measured clean O(N²): ~3.2 s of CPU on a 32 KB input,
   and it burns in a thread the router's timeout cannot cancel.
2. **[F2 · High] Arbitrary code execution in `export_litellm` / `export_guardrails_ai`** —
   `policy.name` is interpolated unescaped into a generated Python module's docstring;
   a crafted name runs code when the module is imported (PoC confirmed).
3. **[F3 · High] The central `EvaluatedPolicy` type is erased to `Any`** and read with
   `getattr(policy, "field", default)` across selection / constraints / explanation
   (28 sites) — the exact logic that chooses recommendations is invisible to mypy, and a
   field rename degrades silently instead of failing.
4. **[F9 · High] Per-candidate recomputation of threshold-invariant quantities** (latency,
   cost, guardrail masks, ID lists) dominates search time and memory; ~30 % of run time is
   recoverable by memoising on `enabled_names` alone, provably without changing results.
5. **[F16 · Medium-High] No CI.** `make check` is manual and already drifting (stale test
   badge); the highest-value safety invariant in the runtime — "in-flight requests finish
   under the policy they started with" — has no regression test.

---

## 2. Architecture

### What the package does

Given guardrail **scores your detectors already produced on labelled traffic**, guardopt
searches the space of guardrail on/off decisions, blocking thresholds, and (optionally)
staged cascade structures, and returns up to three defensible policies — Minimal
(precision-leaning, F0.5), Balanced (F1), Strict (recall-leaning, F2) — each with a
confusion matrix, the case IDs behind every cell, Wilson confidence intervals, a written
explanation, and a committable `Policy` artifact the runtime can enforce.

### Major components and dependency direction

```
                        ┌──────────────────────────── pure core (no I/O, no vendor) ────────────────────────────┐
   CSV / JSON ──▶ matrix.ScoreMatrix ──▶ inputs.OptimiserRequest ──▶ optimise.optimise() ─────────────▶ OptimisationResult
   (cli.py)          (from_csv)            (pydantic validation)          │                                 │
                                                                          ▼                                 ▼
   importers.py ─────────────────────────────────────────────▶  search.search_policies         report.render_markdown
   (DeepEval/TruLens)                                             ├─ exhaustive_search           report_html.render_html
                                                                  ├─ beam_search  (bounded)       (escaped, self-contained)
                                                                  └─ stage_search (cascades)
                                                                          │
                                    candidates.py ──▶ build_candidate_space (threshold pruning + behavioural dedup)
                                                                          │
                                     ┌────────────────────────────────────┴───────────────────────────┐
                                     ▼ (spec)                                                           ▼ (fast path)
                            simulation.py / route.py                                              vectorised.py (NumPy)
                            evaluate_guardrail, _aggregate, stage_verdict   ◀── parity tests ──▶  flat/staged_outcome_codes
                                     │                                                                   │
                                     └───────────────▶ metrics / metrics_intervention / latency ◀───────┘
                                                                          │
                                     selection.select_profiles ─▶ warning_bands.apply_warning_ladder ─▶ explain.py
                                     (Pareto frontier + profile fit)      (derived, not searched)        (prose from numbers)

   ┌──────────────── runtime (imports HTTP; nothing in the pure core imports it) ────────────────┐
   materialise.py ──(calls guardrails)──▶ ScoreMatrix         router.GuardrailRouter.run(request) ──▶ RoutedDecision
   protocol.Guardrail / read_guardrail                        (fail-closed; timeout; on_decision hook)
   adapters.py  (HttpJson, Perspective, pii/keyword, Callable)  monitor.DecisionAggregator / shadow.py / retune.py

   ┌──────────────── sentinel (optional adapter; domain never imports it) ─────────┐
   sentinel/scoring.py (urllib POST /validate, read-only)   sentinel/mapping.py ──▶ service.recommend_policies
```

The **flat/staged unification** is the design spine: a flat policy is a one-stage parallel
route, so the same `Policy` type is searched, serialised, and enforced. `Policy.from_candidate`
bridges search→artifact; `route.evaluate_staged_policy_on_case` on a one-stage policy is
proven equal to the flat `simulation.evaluate_policy`.

### Important execution paths (traced)

- **`optimise()`** (`optimise.py:325`): `(split) → search → (constrain) → select → warning
  ladder → explain`, always all four core steps in one function so a caller can't assemble
  a subtly different pipeline. Holdout, constraints, and bootstrap are opt-in honesty layers.
- **Search** (`search.py`): exhaustive when the arithmetically-computed space ≤
  `max_exhaustive_candidates` (default 50k), else bounded beam (one beam per profile,
  shared memoised evaluator); `stage_search` adds every affordable cascade. Every path
  sizes the space *before* enumerating and refuses rather than hangs.
- **Runtime** (`router.py:234` `run()`): walks stages, calling only what a cascade needs,
  converting every failure mode (raise, timeout, junk) into an error reading that never
  passes and never grants an early exit ("fail closed").

### Public API

`guardopt/__init__.py` re-exports the pure path only (analysis types + `optimise`,
`retune`, report renderers). Runtime enforcement (`GuardrailRouter`) and the Sentinel
adapter are reached via submodules by design, so "analyse a spreadsheet" never imports an
HTTP stack. The layering claim ("`domain` never imports `runtime`/`sentinel`") is real and
test-enforced by an AST import scan.

### Architectural strengths

- Clean pure/runtime/vendor separation with an enforced dependency direction.
- One `Policy` model for search, artifact, and enforcement — the merge is a generalisation,
  not two models kept in sync.
- Semantic rules ("a gap is never a pass", "undefined is `None` not `0.0`", "never infer
  score direction") are enforced structurally (types, validators, refusals), not by
  convention, and are documented at the point of enforcement.
- Reproducibility: deterministic candidate order, seeded splits/bootstraps, arithmetic
  space-sizing.

### Architectural weaknesses

- The recommendation-deciding surfaces (`selection`, `constraints`, `explain`,
  `warning_bands`, `stability`, `monitor`) operate on `Any` and read fields with
  `getattr(…, default)`, so the type system protects none of the logic that picks the
  answer (**F3**).
- The evaluation semantics live in three implementations tied together only by tests, and
  the runtime router hand-rolls a fourth copy of the staged walk (**F4**).
- `search.py` mixes ~5 responsibilities and owns the most widely-consumed domain types
  (**F6**).

---

## 3. Baseline checks

All commands run in the repo's configured venv (`.venv311`, Python 3.11.13, NumPy 2.4.6,
pydantic 2.13.4). No dependencies were installed and no source files were modified.

| Command | Result | Notes |
| --- | --- | --- |
| `.venv311/bin/python -m pytest -q` | **860 passed** in ~18 s | Seeded, deterministic; no network. README badge says "852 passing" — stale (**F16**). |
| `.venv311/bin/ruff check src tests` | **All checks passed** | Rule set deliberately narrow (`E4,E7,E9,F`); documented. |
| `.venv311/bin/mypy` | **Success: no issues in 60 files** | Config is lenient by design; see **F3** for what it therefore misses. |
| `python -m coverage` | **not installed** | No coverage tooling configured; not installed per instructions. Branch gaps (§6) are consequently invisible. |
| `pip-audit` / `bandit` | **not present** | Neither is configured; no security tooling in the repo. |
| CI config | **absent** | No `.github/`, no workflow files. `make check` is the only gate, run manually (**F16**). |
| Lockfile | **absent** | `pydantic>=2.7`, `numpy>=1.26`, no upper bounds, no lock (**F17**). |

The suite, ruff, and mypy are all green; the findings below are not tool warnings but
issues found by reading and tracing the code (plus targeted measurement for the
performance and ReDoS items).

---

## 4. Detailed findings

Ordered by severity, then confidence. IDs are referenced by the remediation plan (§7).

---

### [F1] Quadratic ReDoS in the shipped email PII regex, reachable from live request text

> **Status:** ✅ Fixed — email pattern rewritten to a bounded/possessive linear form and a
> `max_scan_chars` cap added to `HeuristicGuardrail` (`runtime/adapters.py`,
> `runtime/protocol.py`). Measured linear; ~1 MB attack now ~45 ms. Guarded by test T6.

**Category:** Security (denial of service) · **Severity:** High · **Confidence:** High ·
**Effort:** Small
**Location:** `src/guardopt/runtime/adapters.py:152` (`_PII_PATTERNS`), reached via
`pii_guardrail()` → `HeuristicGuardrail.evaluate` (`src/guardopt/runtime/protocol.py:137-146`).

**Observation.** The email pattern is
`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`. `pii_guardrail()` is the package's own
recommended free PII screen; as a `HeuristicGuardrail` it runs `pattern.search(text)` where
`text = str(request.get("text", ""))` — in the runtime, that request text is live,
externally supplied traffic.

**Why it matters.** The two adjacent quantified classes `[A-Za-z0-9.-]+` and `\.[A-Za-z]{2,}`
both match `.` and letters, so a subject that reaches `@` but never a valid TLD forces the
engine to backtrack the `+` one character at a time, at every start offset — polynomial
(quadratic) blow-up. A single ~100 KB crafted request pins a CPU core for tens of seconds;
a modest request rate saturates all cores. The router's `timeout_ms` does **not** contain
it: the regex runs in a worker thread that cannot be cancelled mid-match, so the budget
bounds when the request *returns*, not the CPU burn — which is exactly the fuel for **F5**.

**Evidence (measured, repo venv, this exact compiled pattern).**

| input | 2 000 | 4 000 | 8 000 | 16 000 |
| --- | --- | --- | --- | --- |
| `"a@" + "."*N` | 12.6 ms | 49.9 ms | 198.5 ms | 793.8 ms |
| `"a@" + "a."*N` | 49.5 ms | 197.0 ms | 803.4 ms | 3 213.6 ms |

Clean 4× per doubling = O(N²). Reproduced end-to-end through
`pii_guardrail().evaluate({"text": "a@" + "."*16000})` → 807 ms. A benign 240 KB string is
7 ms, confirming the cost is backtracking, not length.

**Recommendation.** Replace with a linear pattern using disjoint classes and a grouped
label, e.g. `[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+`, and cap `text` length in
`HeuristicGuardrail.evaluate` before scanning (e.g. scan the first N KB, or reject
over-long inputs to a screen). Add a regression test that asserts the screen returns within
a small time bound on a pathological input.

**Trade-offs.** The replacement is marginally less strict about TLD shape; for a
severity-scoring screen (explicitly "a starting screen, not a compliance tool") that is an
acceptable trade for a linear guarantee. A length cap changes behaviour on very long
legitimate inputs — state it in the screen's docs.

---

### [F2] Code injection / RCE in the code-generating exporters via unescaped `policy.name`

> **Status:** ✅ Fixed — `safe_for_generated_source()` (`integrations/common.py`) applied to
> `policy.name`, `guard_names` and `validator_name` in both exporters. The PoC payload is now
> inert text inside the docstring. Guarded by an AST-based injection test.

**Category:** Security (code injection) · **Severity:** High (conditional on name
provenance) · **Confidence:** High (PoC executed) · **Effort:** Small
**Location:** `src/guardopt/integrations/litellm.py:175` and
`src/guardopt/integrations/guardrails_ai.py:111`
(`_MODULE_TEMPLATE.format(policy_name=policy.name, guard_names=…, …)`); templates at
`litellm.py:44-46` and `guardrails_ai.py:22-27`.

**Observation.** Both exporters generate a Python **module** and interpolate `policy.name`
raw into that module's `"""…"""` docstring, and `", ".join(policy.enabled_names)` raw into a
string literal. `policy.name` is `str(payload["name"])` with no charset restriction
(`policy.py:325`); guardrail names are validated only as non-empty. The generated file is
written next to the proxy and imported/executed — the documented workflow.

**Why it matters.** A policy name containing `"""` closes the docstring; whatever follows
becomes top-level module code that runs at import time — arbitrary code execution inside the
LiteLLM proxy or Guardrails-AI host. The embedded `POLICY_JSON` is `json.dumps`-escaped and
is *not* the vector (`"` becomes `\"`, so no run of three quotes survives); the **docstring
name** is the clean break-out.

**Evidence (PoC, run in the repo venv).** Exporting a policy named
`x"""\nimport os\nprint("INJECTED-CODE-RAN", os.getpid())\n_ = """` produced a module whose
first lines are:

```python
"""guardopt policy 'x"""
import os
print("INJECTED-CODE-RAN", os.getpid())
_ = """', compiled into a LiteLLM CustomGuardrail.
```

`exec(compile(module_src, …))` printed `INJECTED-CODE-RAN 91407`. The `POLICY_JSON` block in
the same output shows the name correctly escaped (`"x\"\"\"…"`), confirming the docstring —
not the JSON — is the injection point. `definitions_literal` (`common.py:43`) already uses
`!r` for names and is safe; the exporters are inconsistent with their own helper.

**Recommendation.** Repr/escape the interpolated `policy_name` and `guard_names` (as
`definitions_literal` already does with `!r`), or validate policy/guardrail names against a
safe charset at the model boundary. Note the package already has `sentinel/mapping.slugify`
(`mapping.py:50`) used for exactly this on the Sentinel path — the exporters simply don't
apply the same discipline. A defence-in-depth option is to reject control characters and
quotes in `Policy.name` / `GuardrailBinding.name` in `__post_init__`, since names also flow
into HTML (escaped), JSON (escaped), and now generated code (unescaped).

**Trade-offs.** In the single-user "export my own optimiser output" flow, names are safe
constants and there is no attacker — hence "conditional". The realistic exposure is any
flow where a `Policy` (or its guardrail names) originates from another team, a third-party
policy file loaded via `Policy.from_file`, or tenant-supplied names — which is precisely
what these "hand this artifact to another system" exporters exist for. Fix cost is trivial;
apply it regardless.

---

### [F3] The central result type is erased to `Any` and read with `getattr(…, default)` on the decision path

> **Status:** ✅ Fixed — read-only `RankablePolicy` Protocol added in `domain/evaluation.py`;
> all 28 `getattr(policy, …, default)` reads across selection/constraints/explain/
> warning_bands/stability/monitor/optimise replaced with typed attribute access. mypy now
> checks the ranking logic. (`pareto.py`'s deliberate structural `getattr` left as-is — §8.)

**Category:** Maintainability / type safety · **Severity:** High · **Confidence:** High ·
**Effort:** Medium
**Location:** `selection.py:48-160,205-268`; `constraints.py:95-193`; `explain.py:157-475`;
`warning_bands.py:152-217`; `stability.py:78-111`; `monitor.py:28-41`. 28 `getattr(policy,
"field", default)` reads across these; the object is always the concrete
`EvaluatedPolicy` dataclass (`search.py:180`).

**Observation.** `EvaluatedPolicy` is the single load-bearing object of the optimiser, yet
the entire selection/constraint/explanation chain types it as `Any` (`policy: Any`,
`Sequence[Any]`) and reads its fields defensively, e.g. `getattr(policy, "estimated_cost",
None)`, `getattr(policy, "outcome_signature", None)`, `getattr(policy, "enabled_count", 0)`.
`select_profiles` has exactly one production caller (`optimise.py:388`), which always passes
`EvaluatedPolicy`; the "caller may supply its own comparable object" extensibility the
`getattr` defaults exist for has no production instance.

**Why it matters.** mypy cannot check any attribute access on `Any`, so the ranking and
explanation logic that decides which policy a user ships is entirely unchecked. Worse, the
`getattr`-with-default idiom converts a rename or refactor into a **silent behavioural
change** instead of an error:
- rename `estimated_cost` → every policy silently reads `None` = "price unknown", quietly
  defeating the documented "unknown price is never free / cost sums" invariant and changing
  every cost tie-break;
- rename `outcome_signature` → `deduplicate_by_behaviour` (`selection.py:139`) silently
  treats everything as unique-passthrough, reintroducing the exact "fabricated third
  option" bug the module was written to kill.
These are the codebase's core guarantees, guarded here only by runtime defaults that fail
open.

**Evidence.** `grep` counts 28 `getattr((policy|candidate|p),…)` sites; `constraints.py`
alone has 11 (lines 102-118, 145-186). All operate on `EvaluatedPolicy`, whose fields all
exist — the defaults are dead except under the untested duck-typed path.

**Recommendation.** Type these functions on `EvaluatedPolicy` directly, or define a narrow
`Protocol` (e.g. `RankablePolicy`) capturing the ~12 fields the ranking reads, and access
attributes normally — deleting the `getattr(…, default)` fallbacks. mypy then flags any
rename or missing field at the call site.

**Trade-offs.** Loses the untested extension point; a `Protocol` preserves it *with* static
checking. Mechanical change across ~6 files; no behavioural change intended, and the parity
of behaviour is easy to assert with the existing selection tests.

---

### [F4] Evaluation semantics implemented three times; the runtime router hand-rolls a fourth staged walk

> **Status:** ✅ Fixed — the stop/continue rule is now one function, `route.apply_stage_transition`,
> driven by both the offline walk (`evaluate_staged_policy_on_case`) and the runtime router
> (`router.run`). The NumPy path stays separate but parity-pinned.

**Category:** Architecture (duplication / drift risk) · **Severity:** Medium ·
**Confidence:** High (duplication verified; "will drift" is a risk) · **Effort:** Medium
**Location:** pure per-case (`simulation.py:97-277`, `route.py:33-142`); NumPy
(`vectorised.py:133-270`); runtime (`router.py:252-323`). The FAIL > WARNING > ERROR > PASS
collapse is written four times: `simulation._aggregate` (256), `route.stage_verdict` (33),
and inline in `vectorised.flat_outcome_codes` (189) and `staged_outcome_codes` (244-263).

**Observation.** Three code paths must produce identical verdicts/latency/cost. Two (pure,
NumPy) are held equal by strong randomised parity tests. The third — the live router's
staged state machine (`router.run`) — is a structural re-implementation of
`route.evaluate_staged_policy_on_case` (same `uncertain` / `allow_exit` /
`resolves_uncertainty` / `ON_UNCERTAIN` transitions), sharing only `evaluate_guardrail` and
`stage_verdict`.

**Why it matters.** A concrete cross-path seam already exists inside `optimise()`: holdout
metrics are computed on the **pure** path (`optimise.py:208,214`) while the train metrics on
`EvaluatedPolicy` come from the **NumPy** path (`search.py:337`), and `_holdout_limitation`
(`optimise.py:233`) compares the two to fire the ">10-point overfit" warning — so any
pure/NumPy divergence would spuriously trigger or suppress a *safety* warning. The router
being a fourth hand-rolled walk means a cascade could be enforced differently from how it
was measured, which is the single failure the flat/staged merge exists to prevent. Today
this is mitigated (a router-vs-simulation verdict cross-check test exists), but by tests,
not by shared code.

**Evidence.** `router.run` lines 252-323 duplicate the transition logic of `route.py`
lines 96-129; `stage_verdict` is the only shared helper. The pure/NumPy parity tests
(`test_vectorised_parity.py`, `test_stage_parity_sweep.py`) do not cover the router walk.

**Recommendation.** Extract the stage-transition logic (given per-stage guardrail outcomes,
update `uncertain`/`decided`/`final` and decide exit) into one function that both `route.py`
and `router.py` drive; the router keeps its distinct "decide what to call next" loop but
delegates the verdict transitions. Treat the pure↔NumPy relationship as a derived contract
(the parity test is the contract), and collapse the flat `_aggregate` onto `stage_verdict`.

**Trade-offs.** The NumPy path genuinely can't be auto-derived cheaply and must stay; realistic
scope is unifying the two Python walks and the router. Keep the search hot path untouched.

---

### [F5] Process-global sync-guardrail thread pool with non-reclaimable timed-out calls → cross-router DoS

> **Status:** ✅ Fixed — `executor=` is now injectable on `GuardrailRouter`, `read_guardrail`
> and `materialise`/`materialise_sync`; the shared global is only the default. A caller can
> give a router a bounded, disposable pool so one hung guardrail can't starve every other.

**Category:** Security / resource lifecycle · **Severity:** Medium · **Confidence:** Medium
(mechanism verified; impact deployment-dependent) · **Effort:** Medium
**Location:** `runtime/protocol.py:71` (`_SYNC_GUARDRAIL_EXECUTOR`, module global);
`router.py:346-388` (`_read_contained` abandons timed-out tasks).

**Observation.** Synchronous guardrails run in a single module-global
`ThreadPoolExecutor` (default `max_workers = min(32, cpu+4)`), shared by every
`GuardrailRouter` and every `materialise` in the process. On timeout the router cancels the
asyncio task but, as its own docstring states, cannot cancel the underlying thread (a sync
call blocked in a socket read or a CPU-bound regex) — so a timed-out call frees the *request*
but not the *worker*.

**Why it matters.** An attacker who inflates sync-guardrail latency — trivially via **F1**'s
ReDoS, or a slow/hung upstream behind a sync `HttpJsonGuardrail`/`PerspectiveGuardrail` —
permanently occupies pool workers. Once ~`min(32, cpu+4)` are stuck, *all* sync guardrail
evaluation across *every* router in the process stalls (new calls queue in the executor's
unbounded work queue, which also grows memory). This crosses router/tenant boundaries: one
router's crafted traffic denies unrelated routers sharing the process. Per-request latency
stays bounded (fail-safe WARNING), but throughput collapses and there is no saturation
signal, and an embedder cannot size or dispose the pool.

**Evidence.** `protocol.py:71` constructs the global pool with no `max_workers` argument and
no `shutdown`/`atexit`. `router._read_contained` uses `asyncio.wait` + `task.cancel()` +
`_discard_abandoned_result` (`router.py:41`) — confirming the thread is abandoned, not
joined. The design rationale (a dedicated pool avoids `asyncio.run` teardown joining a
stuck thread) is sound and worth keeping.

**Recommendation.** Make the executor injectable per `GuardrailRouter` (default a bounded,
named pool with a `close()`); cap concurrent in-flight sync calls per router so one router
cannot starve a shared pool; expose the saturation ceiling as a metric and document it. For
truly untrusted CPU-bound guardrails, a `ProcessPoolExecutor` with hard cancellation is the
only way to reclaim the work. Combine with **F1**'s length cap.

**Trade-offs.** Slightly more wiring for callers who currently get a zero-config global;
process pools cost IPC/serialisation. The finding is partly acknowledged in the docstrings —
the gap is that acknowledgement without a bound, backpressure, or signal is still a
production availability risk.

---

### [F6] The optimiser's central result types live inside the 905-line `search.py`

> **Status:** ✅ Fixed — `EvaluatedPolicy` and `SearchDiagnostics` moved to a new
> `domain/evaluation.py`; the two evaluators now share one `assemble_evaluated_policy`
> helper. `search.py` re-exports both names, so existing imports are unchanged.

**Category:** Architecture (cohesion / coupling) · **Severity:** Medium · **Confidence:**
High · **Effort:** Small–Medium
**Location:** `search.py` (905 lines): `EvaluatedPolicy` (180-278) and `SearchDiagnostics`
(281-295), plus space-sizing, `PolicyEvaluator`, `StagedPolicyEvaluator`, and three search
algorithms with their seed/neighbour helpers.

**Observation.** The most widely-consumed domain type (`EvaluatedPolicy`, imported by
`optimise`, `service`, and referenced-as-`Any` by selection/warning_bands) is defined inside
the largest, highest-churn module, which also mixes ~5 responsibilities. The two evaluator
classes independently assemble the same 15-field `EvaluatedPolicy` (`search.py:344` and
`685`) — a copy-paste that must be kept in lockstep by hand.

**Why it matters.** Touching a search algorithm forces recompile/retest of every consumer of
a result; a public entry point (`service.py:22`) reaches into `domain.search` for its return
type; and the duplicated `EvaluatedPolicy` construction is a latent drift point (a new field
added to one evaluator and forgotten in the other yields inconsistent results with no error).

**Recommendation.** Move `EvaluatedPolicy`/`SearchDiagnostics` to a small
`domain/evaluation.py`; factor the shared `EvaluatedPolicy` assembly into one helper both
evaluators call; optionally split the two evaluator classes from the three algorithms.

**Trade-offs.** Import churn only; no behavioural change. Low risk, high readability payoff.

---

### [F7] `retune` re-invokes the private `_split_for_holdout`, depending on undocumented cross-call determinism

> **Status:** ✅ Fixed — `split_for_holdout` is now public with an explicit determinism
> contract in its docstring; `optimise` and `retune` both call it, and a `_split_for_holdout`
> alias is kept for back-compat.

**Category:** Architecture / correctness (latent) · **Severity:** Medium · **Confidence:**
High · **Effort:** Small
**Location:** `retune.py:35,113` imports `_split_for_holdout` from `optimise.py` and calls
it to recompute `holdout_cases`, separately from `optimise(request)` (`retune.py:111`),
which splits internally.

**Observation.** `retune` promotes/keeps a policy on the holdout split. It obtains that
split by calling the *private* `_split_for_holdout` a second time, relying on it being
byte-identical to the split `optimise()` performed internally (same `holdout_seed`).

**Why it matters.** Correctness rests on an invariant that is neither expressed in a type nor
asserted: if `optimise` ever caches the split, changes seeding, or the split becomes
stateful, `retune` would silently judge promote/keep on a *different* holdout than the one
`optimise` searched against and reported — a safety-relevant divergence with no error. Today
it is correct (the split is deterministic), but the coupling is fragile and reaches through
a private symbol.

**Recommendation.** Have `optimise` return (or expose via a public function) the split it
used, and thread the holdout cases into `retune` rather than recomputing them. Alternatively,
make the split a first-class value passed explicitly.

**Trade-offs.** A small signature/return change; removes an import of a private symbol and a
whole class of silent-divergence risk.

---

### [F8] No package exception base; the hierarchy splits arbitrarily across `ValueError` and `RuntimeError`

> **Status:** ✅ Fixed — `domain/errors.py` adds `GuardoptError`, `GuardoptInputError` and
> `SearchSpaceError` (shared parent of both "too large" errors). The six named exceptions are
> reparented, each keeping its original `ValueError`/`RuntimeError` base so existing catches
> still work. Exported at the top level.

**Category:** Maintainability (error handling) · **Severity:** Medium (low urgency) ·
**Confidence:** High · **Effort:** Small–Medium
**Location:** `ValueError` subclasses: `simulation.InvalidThresholdError` (36),
`sentinel/mapping.PolicyNotExpressibleError` (105), `integrations/common.IntegrationExportError`
(15). `RuntimeError` subclasses: `search.CandidateSpaceTooLargeError` (67),
`stage_plans.StagePlanSpaceTooLargeError` (41), `sentinel/scoring.Sentinel*Error` (52-60).

**Observation.** There is no `GuardoptError` base. A consumer cannot separate guardopt's own
rejections from incidental `ValueError`s (e.g. a `float()` parse deep in a loader). The two
"search space too large" errors are the same category yet share no base, so "catch any
search overflow" is impossible. The `ValueError`-vs-`RuntimeError` choice appears incidental.

**Why it matters.** Library ergonomics and forward-compatibility: `except GuardoptError` (and
`except SearchSpaceError`) should be expressible; today only broad, over-catching
`except ValueError`/`except RuntimeError` are available.

**Recommendation.** Introduce `GuardoptError(Exception)` and a couple of category bases
(`GuardoptInputError`, `SearchSpaceError` as the shared parent of both "too large" errors);
reparent existing named exceptions, keeping `ValueError`/`RuntimeError` as secondary bases to
preserve existing `except` call sites.

**Trade-offs.** Minor churn; multiple inheritance to preserve back-compat is slightly
unusual but well-understood.

---

### [F9] Threshold-invariant quantities recomputed for every candidate (latency, cost)

> **Status:** ✅ Fixed — `PolicyEvaluator` memoises latency stats and cost on
> `candidate.enabled_names`. The single biggest contributor to the 27.6 s → 9.3 s speedup.

**Category:** Performance · **Severity:** High (confirmed, provably-equivalent fix) ·
**Confidence:** Very High (measured) · **Effort:** Small
**Location:** `search.py:342,367` (`PolicyEvaluator._latency_stats`) and `search.py:378`
(`_cost_for`), via `vectorised.flat_latency_arrays`/`summarise_latency_arrays` (which sorts
per candidate) and `fanout.latency_by_call_group`.

**Observation.** `_latency_stats` and `_cost_for` take only `candidate.enabled_names` — **no
thresholds**. The mean/p50/p95/p99/timed-count and the summed cost are pure functions of
*which* guardrails are enabled, so with 4 optional guardrails there are ≤ 2⁴ = 16 distinct
results, yet the full pipeline (call-group regrouping + per-case max + `np.sort` +
percentiles) runs for every candidate.

**Why it matters / evidence (measured: 500 cases, 5 guardrails, 228,096-policy exhaustive
space; 27.6 s wall, 2.64 GB peak).** Over a 20k-candidate sample, `_latency_stats` took
0.734 s vs 0.005 s memoised on `enabled_names` (**147×**; only 8 distinct enabled-sets
appeared) — extrapolating to ~8.3 s of the 27.6 s run (~30 %). `_cost_for` is the same
class (~0.9 s). The default 50k-space case routes to beam search, where the identical
per-candidate waste applies.

**Recommendation.** Memoise both on `candidate.enabled_names` (a dict on the evaluator). The
key is already the only input, so results are provably identical.

**Trade-offs.** A dict with ≤ 2^(optional guardrails) entries — negligible memory.

---

### [F10] Per-case ID lists materialised for every candidate, needed only for the reported few

> **Status:** ✅ Fixed — `binary_report_from_codes`/`intervention_report_from_codes` take a
> `with_ids` flag; the search runs `with_ids=False`, and `optimise` rebuilds the ID lists for
> the recommended policies alone via the new public `populate_case_ids`. ~half the
> per-candidate ID work and memory removed.

**Category:** Performance (time + memory) · **Severity:** Medium · **Confidence:** High
(measured) · **Effort:** Medium
**Location:** `vectorised.py:295-299` (`binary_report_from_codes`, 5 `tuple(ids[mask])`) and
`vectorised.py:346-348` (`intervention_report_from_codes`, 3 more).

**Observation.** Each `EvaluatedPolicy` stores the TP/FP/TN/FN/excluded and warned/errored
**case-ID tuples**. Selection and the Pareto filter read only the confusion-matrix counts
and rates; the ID lists are consumed only by the explanations of the ~3 final picks (and the
`frontier`). They are nonetheless built for all ~228k candidates.

**Why it matters / evidence.** The ID-tuple work is ~half of `binary_report`: 0.258 s vs
0.130 s (counts only) over 20k candidates (~2.7 s over the full run), plus the three
intervention tuples. Memory: the eight tuples are ~5.8 KB of the measured 11.6 KB retained
per policy — roughly half of the 2.64 GB peak.

**Recommendation.** Compute counts/rates eagerly; make ID lists lazy — store the compact
`codes` array (~500 B) on `EvaluatedPolicy` and recompute masks/IDs only for the frontier
and the selected policies.

**Trade-offs.** Frontier policies recompute masks once (cheap). Net: ~half the per-policy
memory and a few seconds off the run.

---

### [F11] Search cache memory is O(space × n_cases), bounded only by the space cap

> **Status:** ✅ Fixed — `outcome_signature` is now a compact `bytes` of outcome codes
> (`signature_from_codes`), ~7× smaller than the `tuple[str, …]`. `selection`, `stability`,
> `monitor` and `diff` updated to the codes (constants in `domain/evaluation.py`). Combined
> with F10, peak RSS on the 228k workload fell 2.64 GB → 605 MB.

**Category:** Performance (memory) · **Severity:** Medium · **Confidence:** High (measured) ·
**Effort:** Medium
**Location:** `search.py:308` (`PolicyEvaluator._cache`); retained `outcome_signature`
(`search.py:222`, built at `vectorised.signature_from_codes:352`) plus the **F10** tuples.

**Observation.** Every distinct evaluated policy is retained for the whole search (needed for
selection, dedup, and bootstrap). `outcome_signature` is a `tuple[str, …]` of length
n_cases — 500 Python string pointers ≈ 4040 B/policy — which with the ID tuples is ~9.8 KB of
the 11.6 KB retained per policy.

**Why it matters / evidence.** Peak RSS was **2.64 GB** at 228k policies; at the default 50k
cap that is ~580 MB, and it scales linearly in both space size and n_cases — a raised
`max_exhaustive_candidates` or a larger labelled set (the repo ships n=4000 experiment
datasets) can OOM.

**Recommendation.** Combined with **F10** (removes ~50 %), store the signature as
`bytes`/`np.uint8[n_cases]` instead of `tuple[str,…]` — ~4040 B → ~550 B (~7×), and it builds
faster than `tuple(_OUTCOME_NAMES[merged])`.

**Trade-offs.** A `bytes`/array signature is read by `selection.deduplicate_by_behaviour`,
`stability._blocked_vector`, and `stability._resampled_objective` (which compare `entry ==
"fail"` / `"excluded"`) — a cross-module change those three sites must absorb. Worthwhile
given the memory scaling.

---

### [F12] Per-guardrail masks recomputed per candidate instead of once per (guardrail, threshold)

> **Status:** ✅ Fixed — `_guardrail_masks` accepts a search-lifetime cache keyed on
> `(name, thresholds)`, threaded through `flat_outcome_codes`/`staged_outcome_codes` from the
> evaluators. The redundant hot-path `validate_thresholds` now runs only on a cache miss.

**Category:** Performance · **Severity:** Medium · **Confidence:** High (measured) ·
**Effort:** Small
**Location:** `vectorised.py:133` (`_guardrail_masks`), called from `flat_outcome_codes:182`
— ~1.06 M calls for 228k candidates.

**Observation.** `(fail, warn)` for a given `(guardrail_name, thresholds)` is invariant
across every candidate that contains it. The number of distinct `(name, thresholds)` pairs
equals Σ option_counts ≈ 59, yet the masks are recomputed ~1.06 M times.

**Why it matters / evidence.** 0.299 s → 0.173 s over 20k candidates with a
`(name, thresholds)`-keyed mask cache (46 distinct in the sample) ≈ ~1.4 s over the full run.
This also subsumes **the redundant `validate_thresholds` on the hot path** (`vectorised.py:141`
→ `simulation.py:59`, which builds an `f"[{min},{max}]"` span string on every one of the 1.06 M
calls though the thresholds were already validated in `build_candidate_space`; ~0.3 s, ~1 %).

**Recommendation.** Memoise masks keyed on `(name, thresholds)` on the evaluator (bounded by
the pre-built candidate space); build the masks only from the already-validated space, so the
per-call `validate_thresholds` can drop from the hot path.

**Trade-offs.** ~59 small bool arrays held (~30 KB) — trivial.

---

### [F13] `on_decision` audit hook runs uncontained on the request path

> **Status:** ✅ Fixed — the hook is wrapped in `try/except`; an exception is logged at ERROR
> on `guardopt.runtime.router` and swallowed, so a broken observability sink can't fail the
> request it observed. Guarded by test T3.

**Category:** Code quality / resilience · **Severity:** Low–Medium · **Confidence:** High ·
**Effort:** Small
**Location:** `router.py:342-344` (`if self.on_decision is not None: self.on_decision(decision)`),
outside the request's error containment.

**Observation.** The audit/observability hook is called after the decision is built, with no
`try/except`. Its own docstring states the contract ("must not raise — an exception from it
fails the live request it observed").

**Why it matters.** This is the one place on the live path that is *not* fail-closed:
everywhere else a guardrail raising, timing out, or returning junk is contained into an error
reading (`_read_contained`). A misbehaving metrics/logging sink wired into `on_decision`
therefore takes down the very request it was only meant to observe — an availability incident
caused by observability. Given the package's fail-closed philosophy, this asymmetry is a
genuine design inconsistency, not merely a caller-beware footnote.

**Recommendation.** Wrap the hook in `try/except`, log-and-continue (observability must never
fail traffic), and document that hook exceptions are swallowed. At minimum, pin the current
behaviour with a test so any change is deliberate (see §6, T3).

**Trade-offs.** Swallowing hides a broken hook; mitigate by logging the exception. If a
deployment truly wants hook failures to fail the request, make it an opt-in flag rather than
the default.

---

### [F14] Markdown report performs no output escaping on dataset-derived values

> **Status:** ✅ Fixed — `report._code_span` strips backticks and newlines from case IDs so a
> crafted ID can't break out of the inline code span. Guarded by a markdown-injection test.

**Category:** Security (output injection) · **Severity:** Low · **Confidence:** Medium ·
**Effort:** Small
**Location:** `report.py:28-29` (`_ids`, backtick-wrapped case IDs) and `report.py:74-140`
(raw interpolation of guardrail-name/explanation lines).

**Observation.** Unlike `report_html.py` (which `html.escape`s every value, including SVG
text), the Markdown renderer interpolates `test_case_id`s (from the CSV, `matrix.py:188`,
charset-unrestricted) wrapped only in backticks, and other dataset-derived strings raw.

**Why it matters.** A case ID like `` `x`<img src=x onerror=…> `` or `[t](javascript:…)`
breaks out of the backticks and injects markup that executes when the `.md` is rendered by an
HTML-permitting engine (self-hosted wiki, pandoc, mdBook). The backtick wrapping is
presentation, not a security control. GitHub sanitises; many internal renderers don't.

**Recommendation.** Neutralise interpolated dataset values in `report.py` (strip
backticks/HTML-significant characters from IDs, or route them through a Markdown-escaping
helper), matching what `report_html.py` already does.

**Trade-offs.** Escaped IDs render slightly less cleanly; correctness over cosmetics.
Severity is Low because it depends on both attacker-influenced IDs and a raw-HTML renderer.

---

### [F15] Staged search recomputes invariants inside its inner loop, violating a documented caching invariant

> **Status:** ✅ Fixed — `stage_search` binds `guardrail_by_name` once above the loop, and
> per-stage latency is memoised on `(frozenset(names), parallel)` via a cache threaded
> through `route_latency_arrays`.

**Category:** Performance (staged path) · **Severity:** Low · **Confidence:** High (measured)
· **Effort:** Small
**Location:** `search.py:844-849` — `request.guardrail_by_name[name].score_direction` and
`…call_group` evaluated inside the per-binding genexpr, inside `for combination in
product(...)`. Related: `stage_search` also calls `build_candidate_space(request)` a second
time (`search.py:796`) after the flat search already built it, and rebuilds per-stage latency
per policy (`vectorised.route_latency_arrays` → `stage_latency_arrays`, ~22 % of a staged run).

**Observation.** `guardrail_by_name` (`inputs.py:286`) is deliberately uncached, and its
docstring justifies that by claiming "the two callers in `search.py` each read it once per
search." In `stage_search` it is in fact read ~356k times (measured, 3-guardrail / 60k-policy
run) — the property build re-runs the dict comprehension twice per binding per policy.

**Why it matters / evidence.** Modest in absolute time (~0.14 s at 3 guardrails, scaling with
guardrail and policy count), but it flatly contradicts the invariant the design leans on to
justify not caching, and the `StagedPolicyEvaluator` already holds `self._definitions` for
exactly this. Per-stage latency memoised on `(frozenset(names), parallel)` removes the ~22 %
route-latency cost.

**Recommendation.** Bind `definitions = request.guardrail_by_name` once above the loop (reuse
`self._definitions`); compute the candidate space once and pass it through; memoise
`stage_latency_arrays` on `(frozenset(names), parallel)`.

**Trade-offs.** None — pure refactors, no semantic change. Lower priority because
`search_stages` is off by default.

---

### [F16] No CI; the only quality gate is manual and already drifting

> **Status:** ⛔ Not fixed — **excluded by the maintainer's request.** Adding a CI runner is a
> repository/infrastructure decision rather than a code change. The stale badge is left as-is
> because it is the symptom of this finding, not a separate one. (For reference the suite is
> now 869 tests.)

**Category:** Engineering practice · **Severity:** Medium–High · **Confidence:** High ·
**Effort:** Small
**Location:** repo root (no `.github/`, no workflows); `Makefile` `check: lint types test`;
`README.md:9` badge "tests-852 passing".

**Observation.** `make check` (ruff + mypy + pytest) is the stated contract, run manually.
Nothing blocks a red push/PR. Corroborating drift: the README advertises **852** passing
while the suite is **860** — a stale, hand-maintained number.

**Why it matters.** For a package whose entire value proposition is verified safety
invariants (fail-closed runtime, pure/NumPy parity), the absence of an enforced gate is the
top engineering-practice risk: a regression that a contributor forgets to run `make check`
against ships green. The stale badge is the visible symptom.

**Recommendation.** Add a minimal GitHub Actions job running `make check` on the 3.11–3.13
matrix the classifiers already advertise; regenerate the badge from CI (or drop the hard
count).

**Trade-offs.** Small setup and per-push minutes; standard for a library at this maturity.

---

### [F17] No dependency lockfile / no upper bounds on the runtime deps that carry the hot path

> **Status:** ✅ Partially fixed — conservative `<3` upper caps added to `numpy` and
> `pydantic` in `pyproject.toml`, matching the discipline the docs deps already use, so a
> future major can't jump in silently. A committed `uv.lock` still wants a network-enabled
> `uv` run, which was not available in this environment.

**Category:** Engineering / supply chain · **Severity:** Medium · **Confidence:** High ·
**Effort:** Small
**Location:** `pyproject.toml:42-45` (`pydantic>=2.7`, `numpy>=1.26`, no caps, no lock).

**Observation.** The runtime deps have lower bounds only. `numpy>=1.26` admits NumPy 2.x
(the venv runs 2.4.6), and NumPy is the parity hot path (`vectorised.py`). Notably the *docs*
deps are thoughtfully capped with rationale (`mkdocs>=1.6,<2`, `mkdocs-material<10`); the
load-bearing runtime deps are not — an inconsistency.

**Why it matters.** A behavioural change in a new NumPy would surface only through the
(un-CI'd) suite; there is no lockfile, hash pinning, or reproducible build for contributors
or auditors. For a library, unpinned lower bounds are normal — but the absence of *any*
reproducible dev environment plus no CI compounds the risk.

**Recommendation.** Commit a `uv.lock` (the Makefile already uses `uv`) for the dev/test
environment, and consider a conservative upper cap or an explicit "tested against NumPy 2.x"
statement. Keep the library's runtime lower bounds permissive.

**Trade-offs.** A lockfile is dev-only and does not constrain downstream installs; low cost,
high reproducibility gain.

---

### [F18] Profile → objective/beta mapping duplicated across three modules

> **Status:** ✅ Fixed — a single `metrics.PROFILE_BETA` (plus `profile_objective`) is the
> source of truth; `selection`, `retune` and `stability` consume it, and `f05`/`f1`/`f2` are
> derived from it.

**Category:** Maintainability (DRY / drift) · **Severity:** Low · **Confidence:** High ·
**Effort:** Small
**Location:** `selection.py:108` (`_OBJECTIVE`: MINIMAL→f05, …), `retune.py:41` (`_OBJECTIVE`:
MINIMAL→f05_of, …), `stability.py:41` (`_PROFILE_BETA`: MINIMAL→0.5, …); plus the `f05/f1/f2`
wrappers in `metrics.py:211-223`.

**Observation.** The single fact "Minimal = F0.5, Balanced = F1, Strict = F2" is restated in
at least three places as three different shapes (objective callables in two, beta constants
in a third).

**Why it matters.** Changing a profile's objective requires finding and updating every copy;
a missed one silently makes `stability`/`retune`/`selection` disagree about what a profile
optimises. Low severity because the mapping rarely changes, but it is a genuine multi-module
source-of-truth split.

**Recommendation.** Centralise a single `PROFILE_BETA` (or `PROFILE_OBJECTIVE`) mapping in
one module (e.g. alongside `RecommendationProfile` or in `metrics`), and derive the callables
from it.

**Trade-offs.** None; a small consolidation.

---

## 5. Cross-cutting issues

Several findings share roots; fixing the root resolves multiple symptoms.

- **Root A — "define the type once, enforce it in the type system."** F3 (`Any` + `getattr`
  defaults), F6 (result types buried in `search.py`), F8 (no exception base), and F18
  (duplicated profile mapping) are all facets of load-bearing facts being expressed
  informally or in multiple places rather than as one checked definition. A `RankablePolicy`
  Protocol + a moved `EvaluatedPolicy` + a single profile mapping + an exception base would
  turn four classes of silent-drift risk into compile-time or single-source guarantees.

- **Root B — "one semantics, one implementation (or a checked contract)."** F4 is the
  architectural version; the pure↔NumPy parity is handled well (a contract enforced by
  tests), but the runtime router's hand-rolled staged walk and the four copies of the
  aggregation rule are not. The same root explains why F7 (retune re-deriving the split) is
  fragile: a computed fact is reproduced rather than shared.

- **Root C — "the search hot loop recomputes what is invariant."** F9, F10, F11, F12, and
  F15 are one performance root: quantities that depend only on *which guardrails are enabled*
  (or on a single `(guardrail, threshold)` pair, or on the dataset) are recomputed per
  candidate. Memoisation keyed on the actual inputs (already available) plus a compact
  signature representation roughly halves both run time (~27.6 s → ~14 s) and memory
  (2.64 GB → ~0.5 GB), with results provably unchanged.

- **Root D — "the edges are softer than the core."** F1, F2, F5, F13, F14 all live on the
  live-traffic / output-generation boundary, where the core's "refuse rather than guess"
  discipline is applied less consistently: an unbounded regex over request text, unescaped
  names into generated code, an unbounded shared thread pool, an uncontained hook, and an
  unescaped Markdown sink. Input-length caps + name sanitisation + output escaping +
  hook containment close the class.

- **Trust-boundary summary.** Externally controlled inputs are: CSV/JSON to the CLI
  (validated, DoS-guarded), policy/guardrail names (safe in HTML/JSON, unsafe in codegen —
  F2), and **live request text** at the runtime (the F1/F5 fuel). The Sentinel/HTTP endpoints
  are operator-configured, not attacker-steerable — no SSRF (see §8).

---

## 6. Highest-value missing tests

The suite is excellent — it tests *system properties* (offline ≡ online, NumPy ≡ pure) and
nearly every test documents the regression it guards. These are the gaps that protect the
most, ordered by value.

**T1 — `reload_policy` under an in-flight request (HIGH).** `run()` and `reload_policy`
document a load-bearing production claim: "in-flight requests finish under the policy they
started with; `run()` reads the policy once at entry." Every reload test
(`test_shadow_reload.py`) is single-threaded swap-then-run. A refactor that read
`self.policy` inside the stage loop instead of binding it once (`router.py:237`) would pass
all 860 tests and silently break the invariant. **Test:** an async guardrail blocking on an
`asyncio.Event`; start `run()` as a task; `reload_policy` to a differently-named policy that
would decide the request differently; release the event; assert `decision.policy_name` is the
*old* name and the outcome matches the *old* policy. Deterministic, no wall-clock. *Protects
the runtime's core rotation-safety guarantee.*

**T2 — abandoned-then-raises timeout path (MEDIUM).** `router._discard_abandoned_result`
(`router.py:41`) exists solely to retrieve-and-drop the exception of a task abandoned after
timeout. The only timeout test uses a guardrail that *returns normally* after oversleeping,
so the branch where the abandoned task *raises* is never taken; removing the callback fails
no test but leaks "exception was never retrieved" warnings into production logs. **Test:** a
guardrail that sleeps past the budget then raises; assert WARNING/timeout outcome and (via
`caplog`) no "never retrieved" warning. *Protects log hygiene under the exact failure the
timeout exists for.*

**T3 — `on_decision` that raises (MEDIUM).** Pins the F13 behaviour so any change is
deliberate. **Test:** wire an `on_decision` that raises; assert current behaviour, then — if
F13 is fixed — assert the request still succeeds and the exception is logged, not propagated.
*Locks a live-path resilience decision.*

**T4 — `materialise` order-preservation under real reordering (LOW–MEDIUM).**
`test_concurrent_records_return_in_dataset_order` uses instant `StubGuardrail`s, so completion
order already equals dataset order — it would pass even if the reordering logic broke. **Test:**
give record 0 the slowest guard (descending sleeps) so completion order ≠ dataset order, then
assert row/label alignment. *This is the clearest "passes even if the behaviour broke" case
in the suite.*

**T5 — widen the NumPy/pure parity generator (LOW).** `_random_world` fixes guardrail count
at 3 and cases at 40. Given how central this gate is, sweeping wider fan-out (5–6 signals) and
deeper cascades is cheap insurance against a parity gap that only appears at scale.

**T6 — ReDoS regression for the PII screen (accompanies F1).** Assert `pii_guardrail()`
returns within a small time bound on a pathological input (`"a@" + "a."*20000`), so a future
pattern edit that reintroduces backtracking fails immediately.

---

## 7. Prioritised remediation plan

### P0 — Fix immediately (security / correctness)

- **Replace the email PII regex + cap input length** — resolves **F1**; add **T6**. Small.
- **Escape/sanitise `policy.name` and `guard_names` in the exporters** (reuse `slugify` /
  `!r`) — resolves **F2**. Small.
- **Contain the `on_decision` hook** (try/except, log-and-continue) — resolves **F13**; add
  **T3**. Small.

### P1 — Fix next (high-value architecture / resilience / performance)

- **Type the ranking path on `EvaluatedPolicy` / a `RankablePolicy` Protocol; delete
  `getattr(…, default)`** — resolves **F3**, contributes to Root A. Medium.
- **Memoise latency/cost on `enabled_names`** — resolves **F9** (~30 % run-time, provably
  equivalent). Small. Then **mask cache on `(name, thresholds)`** (**F12**, subsumes the
  hot-path `validate_thresholds`) and **lazy ID lists** (**F10**). Small–Medium.
- **Bound/inject the sync-guardrail executor + per-router concurrency cap** — resolves
  **F5** (pairs with F1's length cap). Medium.
- **Add CI running `make check` on 3.11–3.13; fix the stale badge** — resolves **F16**. Small.
- **Add the `reload_policy` in-flight test (T1)** — highest-value single test.

### P2 — Improve subsequently

- **Move `EvaluatedPolicy`/`SearchDiagnostics` out of `search.py`; factor the shared
  assembly** — **F6**. Small–Medium.
- **Thread the holdout split out of `optimise` into `retune`** — **F7**. Small.
- **Introduce `GuardoptError` + `SearchSpaceError`** — **F8**. Small–Medium.
- **Unify the staged-walk transition logic across `route.py` and `router.py`** — **F4**.
  Medium.
- **Escape dataset values in the Markdown report** — **F14**. Small.
- **Commit a `uv.lock` / cap NumPy** — **F17**. Small.
- **Centralise the profile→beta mapping** — **F18**. Small.
- **`stage_search` inner-loop cleanups** — **F15**. Small.
- Add tests **T2, T4, T5**.

### Measure first (before committing effort)

- **Compact `bytes`/`uint8` outcome signature** (**F11**): the memory win (~7× on the
  signature) is measured, but it touches `selection`/`stability` comparisons — confirm on
  *your* largest realistic dataset (n_cases and `max_exhaustive_candidates`) that peak RSS is
  actually a constraint before taking the cross-module change. The provided harness reports
  peak RSS per configuration.
- **Whether the default (beam) path needs F10/F11 at all**: exhaustive is where memory peaks
  at 2.6 GB; the default 50k cap routes to beam. Profile a representative *bounded* run to
  decide how much of the memory work is worth it for your typical catalogue size.

---

## 8. Things I would NOT change

These looked potentially problematic but are reasonable given the architecture; changing
them would add cost or risk without benefit.

- **No SSRF, despite `urllib.urlopen` on request data.** Every HTTP endpoint is
  operator-configured (`SentinelScorer.base_url` ← settings; `HttpJsonGuardrail.endpoint` is
  a constructor arg; `PerspectiveGuardrail` uses a hardcoded URL). The attacker-controlled
  *text* travels to a *fixed* endpoint; `_origin_of` only strips a path suffix and cannot be
  steered. No file:// / internal-address path from untrusted input. TLS verification and
  scheme-restricted redirects are `urllib` defaults and nothing disables them. **Leave as is.**
- **`random` for the holdout split and bootstrap.** This is statistical resampling, not a
  security context; `random.Random(seed)` is the correct, reproducible choice. Not "insecure
  randomness." **Leave as is.**
- **The pure spec + NumPy fast path duality (not the router duplication).** Two
  implementations of the search evaluation is a deliberate, well-guarded design (pure is the
  spec, NumPy is the engine, property tests pin them equal). F4 concerns the *runtime
  router's* separate walk, not this pair. **Keep the duality.**
- **`build_candidate_space` / `candidates.py` complexity.** `behaviourally_distinct_pairs`
  and `pairs_that_can_block` are O(pairs × cases), but measured at **0.054 s** on the 500-case
  build — negligible vs ~27 s of evaluation, because the `max_threshold_candidates` cap and
  behavioural dedup keep pairs at ~11–12/guardrail. **Do not optimise.**
- **`Policy`/`Stage`/`GuardrailBinding` construction in `stage_search`.** Measured ~12 % of a
  staged run — secondary to the evaluation and route-latency costs (F15). Not the bottleneck
  the size suggests. **Leave as is** (F15's cache/bind cleanups are the better lever).
- **`guardrail_by_name` intentionally uncached (in the flat path).** The uncached design is
  correct and well-justified (a cached index survives `model_copy` and silently describes the
  old guardrails — a verdict-inverting bug). Only `stage_search`'s inner-loop *misuse* (F15)
  is the problem, not the caching decision.
- **Serial `materialise` default (`max_concurrent_records=1`).** Deliberate rate-limit
  friendliness with an opt-in concurrency knob; sensible default for hitting third-party
  scorers. **Leave as is.**
- **The read-only Sentinel client surface, the strict-key policy loader, the "refuse rather
  than guess" density, the long docstrings, and the v1→v2 migration.** All deliberate,
  test-enforced, and central to the package's safety story. **Keep.**
- **HTML report escaping** is complete (every user value via `html.escape(quote=True)`,
  including SVG text); numeric SVG coordinates are safe. **No change** (only the *Markdown*
  path, F14, needs escaping).
- **The `unit` pytest marker selecting everything.** Documented as carried over verbatim from
  the source backend to keep ported files reviewable. Harmless. **Leave as is.**

---

## 9. Final assessment

**Overall production readiness.** High for the offline optimiser (pure, deterministic,
exhaustively validated, 860 green tests); **conditional** for the runtime, pending the
live-edge fixes. The engineering *discipline* is well above typical for a `0.2.0.dev0`
package; the gaps are specific and mostly small to fix.

- **Largest technical risk:** the four-way duplication of evaluation semantics (F4) combined
  with the untyped ranking path (F3) — the two together mean a refactor can silently change
  what the tool recommends or enforces, and the type system won't catch it. Mitigated today
  by an unusually strong test suite, which is exactly why F16 (no CI to *run* that suite on
  every change) elevates the risk.
- **Largest security risk:** the shipped ReDoS regex (F1), because it is reachable from
  untrusted live traffic through the package's own recommended screen, is not contained by
  the request timeout, and directly feeds the shared-thread-pool DoS (F5). The RCE in the
  exporters (F2) is higher-impact per event but requires attacker-influenced names.
- **Largest maintainability risk:** the `Any` + `getattr(…, default)` recommendation path
  (F3) — it removes static protection from the exact code that decides the answer and turns
  renames into silent behavioural changes.
- **Single highest-leverage improvement:** adopt a `RankablePolicy` Protocol and delete the
  `getattr` defaults (F3). It is mechanical, low-risk, and simultaneously restores mypy
  coverage of the decision path, hardens the anti-"fabricated option" and cost invariants,
  and is the anchor of Root A — the most leverage per unit of effort. Close behind it,
  because it is nearly free and provably safe: memoise the threshold-invariant search work
  (F9), which returns ~30 % of run time for a few lines.

*No source files were modified during this review. Throwaway benchmarks and the RCE PoC live
in the session scratchpad; the repo tree is unchanged apart from this file.*
