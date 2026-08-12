"""The package exception hierarchy.

Every error guardopt raises on purpose descends from `GuardoptError`, so a consumer can
`except GuardoptError` to separate this package's own rejections from an incidental
`ValueError` thrown deep in a loader. The two categories below let a caller be more
specific without catching everything:

    GuardoptInputError   the caller gave us something we will not process (a bad policy,
                         a threshold out of range, a policy a target cannot express)
    SearchSpaceError     the search was refused because enumerating it would hang — the
                         shared parent of the flat and staged "too large" errors, so
                         "any search-space overflow" is catchable as one thing

The named exceptions keep their historical `ValueError`/`RuntimeError` base as a SECOND
parent, so existing `except ValueError:` / `except RuntimeError:` call sites keep working
unchanged. This is deliberate: the new base adds a way to catch, it never removes one.
"""

__all__ = [
    "GuardoptError",
    "GuardoptInputError",
    "SearchSpaceError",
]


class GuardoptError(Exception):
    """Base class for every error guardopt raises deliberately."""


class GuardoptInputError(GuardoptError):
    """The caller supplied something guardopt will not process rather than guess at."""


class SearchSpaceError(GuardoptError):
    """A search was refused because enumerating it would hang rather than finish.

    The shared parent of `CandidateSpaceTooLargeError` (flat) and
    `StagePlanSpaceTooLargeError` (staged): both are the same category — an
    arithmetically-sized space over the configured limit — so "any search-space
    overflow" is one `except SearchSpaceError`.
    """
