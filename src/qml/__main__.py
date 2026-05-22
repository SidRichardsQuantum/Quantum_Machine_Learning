"""Command-line entrypoint for ``python -m qml``."""

from __future__ import annotations

from qml.cli import main

if __name__ == "__main__":
    raise SystemExit(main(prog="python -m qml"))
