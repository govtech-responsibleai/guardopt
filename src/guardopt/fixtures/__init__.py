"""Deterministic synthetic datasets for the guardrail optimiser.

`golden` is a hand-built 12-case table whose expected outcomes were computed on paper
from the scores — never by running the optimiser and recording what it said. That
independence is the only thing that makes it a regression test rather than a snapshot
of current behaviour.

**These datasets prove algorithmic correctness only.** They are not evidence that any
threshold is right for real traffic, and nothing derived from them should be presented
as production calibration.
"""
