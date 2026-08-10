"""Pure optimiser domain.

Every module here must import and run with no HTTP, no database, no network, no
Sentinel access, no LLM call and no frontend. That is what makes the optimiser
testable in isolation — and it is enforced by the unit tests, which import only from
this package.
"""
