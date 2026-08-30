# Changelog

Notable changes to guardopt. The format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/) once 0.2.0 lands. Until then every import
path is provisional, as the README says.

## Unreleased (0.2.0)

The first release under the `guardopt` name: the offline optimiser and the runtime
router, merged on one policy model — plus the hardening and honesty work that merge
demanded.

### Fixed

- `ShadowRouter.on_disagreement` is contained like the router's `on_decision`:
  a raising hook is logged, and the incumbent's decision is returned.
- The JSONL loader refuses a non-boolean `unsafe` (a stringified `"false"` was
  truthy and labelled safe records unsafe) and names the line of a non-object.
- Percentiles are genuinely nearest-rank (ceiling rank). `round()` returned a
  lower order statistic for 142 of the first 300 sample sizes — a p95 covered
  as little as 90.9% of requests.
- `f_beta` is computed from integer counts with one division, so policies with
  the same exact F-score tie and the documented tie-breakers actually run.
  Two rounded floats gave exact 2/3 three different representations.
- `TestCaseGuardrailResults.result_for` follows a replaced `guardrail_results`
  list. `calibrated_cases` produced cases whose cached index still held the raw
  scores, so the pure path (candidates, simulation, holdout, explanations) read
  uncalibrated values while the NumPy search read calibrated ones.
- `make build` cleans first: a stale `build/lib/` was leaking six retired v1
  modules into the wheel.
- Stage search enforces mandatory guardrails: a cascade omitting one can no longer be
  evaluated, selected, or recommended.
- Staged selections survive `optimise()`'s warning ladder with their structure intact.
  A cascade gets no back-derived warning band — a band is not inert inside a cascade —
  and is no longer crashed on or silently flattened.
- Staged searches report themselves truthfully: `SearchMethod.STAGED`, with space and
  evaluation counts covering both the flat and the cascade halves.
- Selection collapses Pareto-identical policies before the O(m²) frontier pass, so the
  "refuse rather than hang" guarantee holds at selection, not only at enumeration.
- Explanations no longer describe a sequential cascade as running "in parallel".
- The router validates at construction that the policy file and the definitions agree on
  `score_direction` (a mismatch silently inverts verdicts) and that every threshold is
  in range — not on the first live request.
- `HttpJsonGuardrail` treats a 200 with an unparseable body the same as a transport
  failure: an error reading, never an exception out of a live request. The except tuple
  now covers `IncompleteRead`, `ConnectionResetError`, and custom-parser failures.
- `materialise` fans a multi-label guardrail's error out to one row per signal, so one
  transient failure no longer produces a matrix `optimise()` rejects; raising guardrails
  become error rows instead of discarding the whole run.
- `constraints.objective()` no longer hands an unmeasured false-positive count the best
  possible score, and `EvaluatedPolicy` gained the `false_positive_rate` the constraints
  API always read.
- Policy files parse strictly: unknown keys are refused naming their JSON path (a
  `"warnning"` typo used to load as a never-flags binding), explicit `"warning": null`
  is refused, and a first stage conditioned `on_uncertain` — which can never run — is
  refused at construction.
- The `py.typed` marker ships, making the `Typing :: Typed` classifier true.

### Added

- **Dataset checks** (`check_dataset`, `domain/sanity.py`), run by `optimise()` on the
  full dataset and leading `result.warnings`: one-label datasets, guardrails with no
  score, a constant score or scores on only a handful of cases, and conflicting labels
  on identical results — reported with `unavoidable_errors`, the floor on false
  positives plus false negatives that no policy can beat. `OptimisationResult.dataset`
  carries the structured report.
- **Non-finite numbers are refused** wherever one enters: scores, ranges, default
  thresholds, latencies, costs and weights on the input contract (a `[nan, 1.0]` range
  used to pass the `min < max` check, because `nan >= 1.0` is `False`); blocking and
  warning thresholds on policy bindings (`json.loads` accepts a bare `NaN`, and a NaN
  line is a binding that never fires); and constraint bars, whose rates must also lie in
  `[0, 1]` and latencies be non-negative — `Constraints(min_recall=98)` is refused
  rather than reported as a bar every policy misses.
- **The NumPy fast path** (`domain/vectorised.py`). The search's evaluation hot loop
  lowered to arrays: 19–40× faster on benchmark configs, identical verdicts. The pure
  per-case path stays in the codebase as the specification, held equal by a randomised
  parity suite; numpy joins pydantic as the core's second dependency.
- **The analysis kit.** `diff_policies` (case-level diff between two policies,
  bucketed by label); `evaluate_slices` (per-slice confusion metrics with Wilson
  intervals, small slices flagged); `calibrate`/`calibrated_cases` (PAV isotonic
  calibration onto a shared P(unsafe) scale); `suggest_labels` +
  `unsafe_labels_needed` (active labelling, ranked and explained);
  `false_negative_bound` (exact one-sided Clopper–Pearson bound, attached to every
  holdout evaluation).
- **The retune loop** (`guardopt.retune`, CLI `guardopt retune`). Optimise on fresh
  traffic and compare against the incumbent on the shared holdout: promote only on a
  strict out-of-sample win, keep otherwise, refuse without a holdout.
- **The HTML report** (`render_html`, CLI `--html`). One self-contained file with the
  whole Pareto frontier as a trade-off chart, profile picks ringed, limitations first,
  everything escaped. `OptimisationResult` now carries `frontier`.
- **Guard adapters** (`runtime.adapters`). A heterogeneous fleet shelf beside
  `HttpJsonGuardrail`: a free SG-flavoured PII regex screen, a keyword blocklist,
  `CallableGuardrail` for local classifiers, and `PerspectiveGuardrail` (failures
  become error readings; the API key is scrubbed from every error path).
- **LiteLLM lifecycle modes.** `export_litellm` emits the matching hook per mode —
  pre_call, during_call, and post_call scanning the model's response — with the block
  surfacing as a clean HTTP 400, verified against a live proxy.
- **Integrations** (`guardopt.integrations`). Exporters that deploy a recommended
  policy where guardrails already run: LiteLLM and Guardrails AI exports wrap the
  policy in each target's custom-guardrail mechanism running guardopt's own router
  (cascades and fail-closed semantics preserved as measured, since neither target
  can express them natively); the OpenAI Guardrails export emits the pipeline
  bundle with the loss stated loudly — the scorer changes, cascades are refused
  rather than flattened. Importers turn DeepEval test runs and TruLens
  records-and-feedback rows into a `ScoreMatrix`, with the direction convention
  stated (never inferred) and unlabelled cases refused by name.

- **Warning bands as the cascade's routing dimension**
  (`OptimiserConfig.search_stage_bands`, on by default with `search_stages`). Non-final
  stages search two-threshold bands: below the band exits early, past the blocking line
  blocks early, and only the ambiguous middle escalates — to a final stage that
  adjudicates (`resolves_uncertainty`), so a cleared escalation is a pass, not a
  residual flag. On the synthetic RQ2 sweep this took cascade savings at matched
  accuracy from under 1% to 12–97%, and in some configurations a banded cascade beats
  the best flat policy's F1 outright — three-way routing reaches behaviours no flat
  threshold policy can express. The banded space is sized before enumeration like
  everything else, and the flag off reproduces the blocking-only search (the ablation).
- **The money half of the cost model, wired to data.** A per-call price reaches the
  optimiser from either side — observed `cost` on result rows (carried through
  `materialise`), or a declared `GuardrailDefinition.cost_per_call` — with measured
  beating declared. `EvaluatedPolicy.estimated_cost` charges per call group (money sums
  where latency takes the max; a cascade pays only for the routes taken), cost joins the
  Pareto frontier as a fourth axis under the latency rules, explanations and the report
  state it, staged cards quote the p95 route latency, and `RouteTrace` reports `None`
  rather than 0.0 when nothing was measured.
- **Failure containment and budgets in the router.** A raising guardrail becomes an
  error reading (never a pass, never an early exit); an optional `timeout_ms` bounds
  every call by abandoning overruns. Sync guardrails run on worker threads, so parallel
  stages genuinely parallelise.
- **Wilson 95% confidence intervals** on every recommendation's recall and precision.
- **Holdout evaluation** (`OptimiserConfig.holdout_fraction`): search on a stratified
  deterministic split, report train and holdout numbers side by side, refuse with a
  named reason when the holdout would be too small to mean anything.
- **Bootstrap stability** (`OptimiserConfig.bootstrap_rounds`): how often each
  recommended policy survives a resampled dataset, recomputed from stored outcome
  signatures with no re-simulation.
- **Constraints, wired in**: `optimise(constraints=Constraints(...))` selects only from
  policies clearing every stated bar, or returns unconstrained results with the misses
  named. Documented on its own page.
- **The committable artifact**: every recommendation carries itself as a `Policy` ready
  for `to_file()`, review, and the router.
- **`ScoreMatrix.from_csv`**: the scores-in-a-spreadsheet path, with gap-is-never-a-pass
  cell semantics and refusal of unknown columns.
- **A `guardopt` CLI** (`guardopt optimise scores.csv --guardrails g.json --out report.md`)
  and a Markdown report renderer with limitations first and case IDs on every
  confusion-matrix cell.
- **Sentinel batch scoring** (`score_dataset`): bounded retry on 429/5xx/timeouts only,
  record-or-raise failure handling, transport errors that keep a capped, credential-free
  response excerpt.
- **Audit and monitoring**: decisions carry the policy name, schema version and
  timestamp; the router takes an `on_decision` hook; `runtime.monitor` compares live
  outcome rates against the simulated baseline, refusing to conclude drift from small
  samples.
- **Shadow mode and hot reload** (`runtime.shadow.ShadowRouter`,
  `GuardrailRouter.reload_policy`): measure a candidate policy on live traffic without
  enforcing it; swap policies atomically with full validation.
- Cascade-aware explanations: stage-by-stage guardrail lines, the measured early-exit
  share, and limitations covering route-varying latency and search convergence.
- Offline/online verdict parity pinned by a generated sweep — representative structures
  crossed with every score band, 500 seeded random cascades, and fanned-out signals.
- Local tooling replacing CI: `make check` (ruff, mypy, pytest), `make docs`,
  `make docs-deploy`.

### Removed

- The GitHub Actions workflows (test matrix and docs deploy). Checks run locally via
  `make check`; the docs site deploys manually.
- (Earlier in the merge) the v1 engine and the v1 policy artifact, per the migration
  guide: v1 files are read by `load_v1` where honest conversion is possible, and refused
  where it is not.
