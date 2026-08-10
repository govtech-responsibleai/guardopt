"""Calling guardrails, and enforcing a policy on live traffic.

Nothing under `guardopt.domain` imports this package. The optimiser works from scores that
already exist and must stay runnable with no network, no credentials and no vendor; this is
where those things live.
"""
