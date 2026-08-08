# Experiments

The paper's evidence lives here. Everything is seeded and runs from committed inputs, so
every number in the paper regenerates from a clean checkout — the scored datasets, once
built, are the only artifact that costs API money to reproduce, which is why they get
released alongside the paper.

## Research questions

- **RQ1** — Does joint multi-guardrail threshold search beat per-guardrail independent
  tuning (sklearn `TunedThresholdClassifierCV`), TruSThresh (WSDM'23, reimplemented), an
  Optuna multi-objective search, and the best single guardrail?
- **RQ2** — When does a cascade beat the best flat policy? Characterised by the cost
  ratio between stages, the score correlation between guardrails, and traffic skew.
  `synthetic_pilot.py` is the controlled-sweep half; the scored real datasets validate.
- **RQ3** — Headline: % cost and latency saved at equal recall.
- **RQ4** — Honesty: the holdout gap and bootstrap stability of every method's picks.
- **RQ5** — Ablations: warning bands on/off, error injection (fail-closed end to end),
  bounded vs exhaustive search quality, mandatory-guardrail constraints.

## Layout

```
experiments/
  README.md            this file
  synthetic_pilot.py   RQ2 controlled sweep on generated data — runs offline, now
  results/             outputs (gitignored; regenerate, don't commit)
```

Planned as the real datasets land (see PAPER-PLAN.md for the full sequencing):

```
  datasets/            loaders + label mapping for ToxicChat, WildGuardMix, OpenAI
                       moderation eval, BeaverTails, Jigsaw, UnSmile
  score.py             the materialise harness: datasets x guardrail fleet -> CSVs,
                       recording real per-call latency and cost
  baselines/           per-guardrail sklearn tuning, TruSThresh reimplementation
                       (from the Wayback snapshot of hyperconnect/trusthresh),
                       Optuna joint search, best-single-guardrail
  run_rq1.py ... run_rq5.py
```

## Running

```bash
.venv311/bin/python experiments/synthetic_pilot.py
```

Prints the sweep table and writes `results/synthetic_pilot.csv`.

## Pilot findings so far

The first pilot run (2026-08-09) surfaced two things worth more than the table itself:

1. **The three profiles never trade accuracy for money.** F1 leads every profile's sort
   key and cost is a distant tie-breaker, so a 10x-cheaper cascade that gives up 0.01 of
   F1 never claims a profile slot. Cost-sensitive selection is a constraints /
   `best_by_objective` question, and the paper should present it that way.
2. **Searched cascades lack the mechanism that makes cascades win.** Non-final stages
   always allow exit, and with no warning bands searched for cascade stages, an exit
   fires on any clean pass — so the ambiguous middle exits instead of escalating, and
   the achievable saving collapses to the cheap guardrail's share of the bill. The
   classic fix (WaldBoost's two-threshold band per stage, routing only the uncertain
   middle onward) is precisely what `stage_search` does not yet search. Searching stage
   bands — or `allow_exit` as a variant — is the method work RQ2 now motivates with
   data.
