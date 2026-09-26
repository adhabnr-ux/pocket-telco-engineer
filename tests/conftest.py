import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "tasks" / "ofdm_channel_estimation"
EVAL_DIR = TASK / "eval"
BASELINES = TASK / "baselines"

# Make the task harness importable as a module (``import eval as ev``) for white-box tests.
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


@pytest.fixture(scope="session")
def task():
    from pte.task import Task

    return Task(TASK)


@pytest.fixture(scope="session")
def ev():
    import eval as ev_module

    return ev_module
