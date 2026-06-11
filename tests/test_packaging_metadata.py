from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def _project_metadata() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_package_uses_src_layout_and_console_script() -> None:
    metadata = _project_metadata()

    assert metadata["project"]["version"] == "0.2.13"
    assert metadata["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert metadata["project"]["scripts"]["qml-pennylane"] == "qml.cli:main"
    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == ["LICENSE"]


def test_python_classifiers_match_ci_matrix() -> None:
    metadata = _project_metadata()
    classifiers = set(metadata["project"]["classifiers"])

    assert "Programming Language :: Python :: 3.10" in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers


def test_project_version_has_matching_top_changelog_entry() -> None:
    metadata = _project_metadata()
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    release_headings = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.MULTILINE)

    assert release_headings
    assert release_headings[0] == metadata["project"]["version"]


def test_generated_result_pages_match_project_version() -> None:
    metadata = _project_metadata()
    version = metadata["project"]["version"]
    result_pages = [
        Path("docs/results/api-reference.md"),
        Path("docs/results/benchmarks.md"),
        Path("docs/results/real-examples.md"),
        Path("docs/results/tutorials.md"),
    ]

    for page in result_pages:
        text = page.read_text(encoding="utf-8")
        assert f"- Package version: `{version}`" in text
