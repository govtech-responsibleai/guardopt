# guardopt — working agreement for agents and humans

`guardopt` measures existing guardrails against labelled traffic and finds the thresholds,
composition and cascade routing to deploy. Read `README.md` for what it is;
`docs/methodology.md` for what the search does and does not prove.

## Layout

| Path | What lives there | Rule |
|---|---|---|
| `src/guardopt/domain/` | The pure optimiser core | **Imports nothing from `runtime`, `integrations`, `sentinel`, or the network.** A test asserts this. |
| `src/guardopt/runtime/` | Router, adapters, materialise, monitor, shadow | Fail-closed: an error reading is never a pass. |
| `src/guardopt/integrations/`, `sentinel/` | Exporters/importers for other systems | Refuse what a target cannot express, by name. |
| `tests/` | 950+ deterministic tests, ~12 s | Run from the repo root — a few tests read `src/` relatively. |
| `docs/` (MkDocs), `web/` (landing page) | The public site | `web/` only when the task *is* the landing page. |
| `experiments/`, `paper/` | Research runners and the draft | `paper/`, `NEXT-STEPS.md`, `PAPER-PLAN.md` are local notes — never commit them. |

## Environment and the gate

```bash
.venv311/bin/python            # the interpreter; do not use .venv/
make check                      # ruff + mypy + pytest — THE merge gate; there is no CI
make docs                       # mkdocs --strict; run when you touched docs/
```

`make check` must be green on the exact tree you push. "Tests pass" means you ran them
after your last edit, not before it.

## Multiple agents: one task, one branch, one worktree

Several agent sessions work on this repo concurrently. The rules that keep their work
mergeable:

1. **Branch from fresh `origin/main`, never from another task's branch.**
   ```bash
   git fetch origin
   git worktree add ../guardopt-<slug> -b <type>/<slug> origin/main
   ```
   A worktree per task means no two agents share a working tree, an index, or a build
   directory. `guardopt-merge` was the integration branch for the initial merge; it is
   retired once its last commit lands on `main` — do not start new work on it.

2. **One concern per branch, small enough to review in one sitting.** If you discover a
   second problem while fixing the first, note it and open a second branch. Do not
   "while I'm here" across modules.

3. **Touch only what the task needs.** No reformatting, no reordering imports, no renaming
   in files you did not need to change. The shared hotspots — `src/guardopt/__init__.py`,
   `src/guardopt/optimise.py`, `CHANGELOG.md`, `docs/*.md` — are where parallel branches
   collide: in those, *append* rather than rewrite, and keep the hunk minimal.

4. **Rebase before you push, then re-run the gate on the rebased tree.**
   ```bash
   git fetch origin && git rebase origin/main && make check
   ```
   Conflicts in `CHANGELOG.md` are expected and resolved by keeping both sides. Conflicts
   in code mean two tasks overlapped — resolve by re-reading the other change, not by
   picking "ours".

5. **Merge through a pull request, "Rebase and merge".** Commits here carry real
   messages (see below), so they survive the merge intact and history stays linear. No
   direct pushes to `main`. The reviewer (human or agent) checks that `make check` is
   green on the rebased head and that the CHANGELOG and docs moved with the code.

6. **Commit when a unit of work is green, not at the end of the session.** Finished,
   tested work sitting uncommitted in a working tree is work the next agent will
   clobber or duplicate.

## Commit messages

`type: a sentence stating the change and the reason`, where type is one of
`feat`, `fix`, `docs`, `polish`, `chore`, `test`. The body says *why* — what was wrong,
what the evidence was, what the alternative was and why it lost. Numbers over
adjectives (`27.6s -> 9.3s`, `952 tests`). End with the co-author trailer:

```
Co-Authored-By: Claude <model name> <noreply@anthropic.com>
```

## Every change carries its paperwork

- **User-visible behaviour** → a bullet under `## Unreleased` in `CHANGELOG.md`.
- **Behaviour the docs describe** → the docs page changes in the same PR. The docs are
  the contract; a doc that describes the previous behaviour is a bug.
- **A refusal or a warning** → a test that asserts on the *message* (`match=`), not just
  that something raised. A validator that rejects the right input for the wrong reason
  is still a bug.
- **A new number** (rate, bound, percentile) → its definition in a docstring, and a test
  against a hand-computed or closed-form oracle.

## The invariants (do not trade these for convenience)

- **Undefined is `None`, never `0.0`.** A policy that blocked nothing has no precision.
- **Never infer a guardrail's score direction.** It is stated metadata.
- **A gap is never a pass.** "Could not check" and "checked, clean" stay distinguishable.
- **Every number is finite.** NaN and inf are refused at the point of entry.
- **Refuse rather than hang or guess**, and name the reason in the message.
- **A bounded search reports that it was bounded.** Best-found is not best-possible.
- **Offline and runtime verdicts are identical.** A change to either side must keep the
  parity tests green; if you change the rule, change it in `apply_stage_transition`.
- **Small samples are flagged, not hidden.**

## Release

Not on any branch other than a deliberate release PR. `pyproject.toml` version,
`CHANGELOG.md` heading, and the two "provisional API" banners (README, `docs/index.md`)
move together. Build only from a clean checkout — a stale `build/` directory leaks
retired modules into the wheel.
