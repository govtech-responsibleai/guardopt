# Changelog

Notable changes to guardopt. The format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/) once 0.2.0 lands. Until then every import
path is provisional, as the README says.

## Unreleased (0.2.0)

The first release under the `guardopt` name: the offline optimiser and the runtime
router, merged on one policy model — plus the hardening and honesty work that merge
demanded.

### Fixed

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
