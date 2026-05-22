from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(PROJECT_ROOT / "src")


def pytest_configure() -> None:
    pythonpath = os.environ.get("PYTHONPATH")
    paths = pythonpath.split(os.pathsep) if pythonpath else []

    if SRC_PATH not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([SRC_PATH, *paths])
