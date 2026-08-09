"""Labelled public datasets, loaded into the shape the scoring harness feeds guardrails.

Every loader returns `LabelledRecord`s — the same type `materialise` consumes — with a
deterministic, stratified subsample so a paper table regenerates byte-identically from
the cached pool. Loaders verify the upstream field names at runtime and refuse, naming
the field and the dataset, when a schema has drifted: a silently remapped label column
would poison every number downstream.

Licences are recorded per loader and must be checked before the scored matrices are
redistributed — that is a release-time decision, not a loader-time one.
"""

from experiments.datasets.loaders import DATASETS, load_dataset

__all__ = ["DATASETS", "load_dataset"]
