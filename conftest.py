"""Rootdir conftest. One job: keep collection away from paths it may not touch.

The sandbox this repo is developed under denies access to secret files (`.env`) hard
enough that even `stat` fails — and pytest's default ignore hook stats every rootdir
entry while deciding what to collect, which turns a permission denial into a collection
error before a single test runs. Answering "ignore it" ahead of the default hook means
the question is never asked.
"""

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    if collection_path.name.startswith(".env"):
        return True
    return None
