"""Stage plans: every ordering of some guardrails, cut into stages — and the arithmetic
that decides whether enumerating them is a search or a hang.

A stage plan is an ordered subset of the guardrails, partitioned into consecutive stages.
The space is the product of three things that each grow fast:

    ordered subsets of n guardrails      sum over k of P(n, k)
    ways to cut an ordering of k         compositions of k into parts <= max_stage_size
    threshold policies                   already a product across guardrails

On five guardrails with stages up to three wide that is 2,685 plans — before a single
threshold is chosen. The threshold space for five guardrails is already around 22,000
policies, so the combined space is roughly sixty million.

So this follows the discipline the threshold search already follows, and for the same
reason: **size it arithmetically, check it, and only then enumerate.** An unguarded product
is not a slow search, it is a hang, and a hang is far harder to diagnose than a refusal.

The counting functions are the guard, so they have to be the truth rather than an estimate.
A test asserts that `enumerate_stage_plans` yields exactly `count_stage_plans` items across
a range of shapes — if those ever disagree, the guard is protecting against the wrong number.
"""

from collections.abc import Iterator, Sequence
from functools import cache
from itertools import permutations

__all__ = [
    "StagePlan",
    "StagePlanSpaceTooLargeError",
    "check_combined_space",
    "count_compositions",
    "count_stage_plans",
    "enumerate_stage_plans",
]

#: An ordered tuple of stages, each an ordered tuple of guardrail names.
StagePlan = tuple[tuple[str, ...], ...]


class StagePlanSpaceTooLargeError(RuntimeError):
    """The combined stage-plan and threshold space exceeds the configured limit.

    Raised *before* enumeration begins, and carries the estimate — a refusal that does not
    quote the number leaves the caller unable to decide whether to raise the limit or
    narrow the problem.
    """

    def __init__(self, estimated_size: int, limit: int) -> None:
        self.estimated_size = estimated_size
        self.limit = limit
        super().__init__(
            f"stage-plan search would evaluate an estimated {estimated_size:,} policies, "
            f"which exceeds the configured limit of {limit:,}. Reduce the guardrail count, "
            f"narrow max_stage_size, or use the bounded search."
        )


@cache
def count_compositions(length: int, max_part: int) -> int:
    """How many ways an ordering of `length` items cuts into stages of at most `max_part`.

    The recurrence is the whole content: the final stage takes 1, 2, ... or `max_part`
    items, and whatever precedes it is the same problem one size down.

        C(0) = 1
        C(k) = C(k-1) + C(k-2) + ... + C(k-max_part)

    With `max_part=2` that is Fibonacci; with 1 it is always a single all-singletons cut.
    """
    if max_part < 1:
        raise ValueError("max_part must be at least 1")
    if length == 0:
        return 1
    return sum(
        count_compositions(length - part, max_part)
        for part in range(1, min(max_part, length) + 1)
    )


def count_stage_plans(guardrail_count: int, max_stage_size: int) -> int:
    """Total stage plans over `guardrail_count` guardrails, by multiplication.

    Sum over subset size k of (orderings of k) x (ways to cut k):

        sum over k of  P(n, k) * C(k)

    Never computed by generating and counting — that is the thing being guarded against.
    """
    total = 0
    permutation_count = 1  # P(n, 0)
    for k in range(1, guardrail_count + 1):
        permutation_count *= guardrail_count - k + 1  # P(n,k) from P(n,k-1)
        total += permutation_count * count_compositions(k, max_stage_size)
    return total


def _cuttings(length: int, max_part: int) -> Iterator[tuple[int, ...]]:
    """Every way to cut `length` items into consecutive runs of at most `max_part`."""
    if length == 0:
        yield ()
        return
    for part in range(1, min(max_part, length) + 1):
        for rest in _cuttings(length - part, max_part):
            yield (part, *rest)


def enumerate_stage_plans(
    guardrail_names: Sequence[str], max_stage_size: int
) -> Iterator[StagePlan]:
    """Every stage plan, in deterministic order.

    Deterministic because the search must be reproducible: the same input has to give the
    same recommendation, or "we measured this" means nothing.

    Ordered by subset size, then by permutation, then by cutting. A single guardrail in a
    single stage is included, so the flat case never falls outside the enumeration.
    """
    if max_stage_size < 1:
        raise ValueError("max_stage_size must be at least 1")

    for k in range(1, len(guardrail_names) + 1):
        for ordering in permutations(guardrail_names, k):
            for cutting in _cuttings(k, max_stage_size):
                stages: list[tuple[str, ...]] = []
                offset = 0
                for part in cutting:
                    stages.append(tuple(ordering[offset : offset + part]))
                    offset += part
                yield tuple(stages)


def check_combined_space(
    *, stage_plan_count: int, threshold_space_size: int, limit: int
) -> int:
    """Refuse the search if stage plans x threshold policies exceeds `limit`.

    The boundary is **inclusive** — exactly at the limit is allowed. Stated because an
    off-by-one here is invisible until somebody's search hangs at precisely the configured
    size. Returns the estimate so a caller can report it on the way through.
    """
    estimated = stage_plan_count * threshold_space_size
    if estimated > limit:
        raise StagePlanSpaceTooLargeError(estimated, limit)
    return estimated
