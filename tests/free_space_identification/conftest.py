import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def baseline_dir() -> Path:
    configured = os.environ.get("BTS_FREE_SPACE_ZBF_DIR")
    if not configured:
        pytest.skip("BTS_FREE_SPACE_ZBF_DIR is not configured")

    path = Path(configured).expanduser().resolve()
    if not path.is_dir():
        pytest.fail(f"BTS_FREE_SPACE_ZBF_DIR is not a directory: {path}")
    return path


@pytest.fixture(scope="session")
def baseline_report() -> Path:
    configured = os.environ.get("BTS_FREE_SPACE_REPORT")
    if not configured:
        pytest.skip("BTS_FREE_SPACE_REPORT is not configured")

    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"BTS_FREE_SPACE_REPORT is not a file: {path}")
    return path
