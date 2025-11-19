import os

import pytest


@pytest.fixture
def index_node() -> str:
    return os.getenv("ESGPULL_INDEX_NODE", "esgf.ceda.ac.uk")
