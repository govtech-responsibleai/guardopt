"""Sentinel-facing types: the policy shape, the verified catalogue, and the mapping.

**Nothing in this package writes to Sentinel.** The client exposes read and validate
methods only; no deploy or update method exists, and adding one is a separate, explicitly
approved change. Every policy this package emits is a draft with `is_deployed: false`.
"""
