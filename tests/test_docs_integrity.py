from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    path = ROOT / "docs/pages/generate_results.py"
    spec = importlib.util.spec_from_file_location("generate_results_for_tests", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook_output_text(notebook) -> str:
    parts: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            parts.append(str(cell.source))
            continue
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = output.get("text")
            if text:
                parts.append("".join(text) if isinstance(text, list) else str(text))
            data = output.get("data", {})
            text_plain = data.get("text/plain")
            if text_plain:
                parts.append(
                    "".join(text_plain) if isinstance(text_plain, list) else str(text_plain)
                )
    return "\n".join(parts)


def test_notebook_result_extraction_does_not_modify_notebooks(monkeypatch, tmp_path) -> None:
    generator = _load_generator()
    notebook_path = ROOT / "notebooks/tutorials/01-classical-vs-quantum-classifier.ipynb"
    before = notebook_path.read_bytes()

    monkeypatch.setattr(generator, "NOTEBOOK_ASSETS", tmp_path / "notebook-results")
    result = generator.collect_notebook_result(notebook_path, "tutorial")

    assert notebook_path.read_bytes() == before
    assert result["streams"]
    assert result["images"]
    assert list(tmp_path.rglob("*.png"))


def test_docs_result_local_links_and_images_point_to_existing_files() -> None:
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

    for page in sorted((ROOT / "docs/results").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for target in [*link_pattern.findall(text), *image_pattern.findall(text)]:
            if re.match(r"^[a-z]+:", target):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part or path_part.endswith(".html"):
                continue
            target_path = (page.parent / path_part).resolve()
            assert target_path.exists(), f"{page}: missing local target {target}"


def test_workflows_parse_and_release_order_is_tests_publish_pages() -> None:
    workflows = {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
    }

    assert workflows["tests.yml"]["name"] == "Tests"
    assert workflows["publish.yml"]["name"] == "Publish"
    assert workflows["pages.yml"]["name"] == "Pages"

    tests_text = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    publish_text = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    pages_text = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")

    assert 'tags:\n      - "v*"' in tests_text
    assert "workflow_run:" in publish_text
    assert "- Tests" in publish_text
    assert "github.event.workflow_run.conclusion == 'success'" in publish_text
    assert "startsWith(github.event.workflow_run.head_branch, 'v')" in publish_text
    assert "workflow_run:" in pages_text
    assert "- Publish" in pages_text


def test_stable_metadata_rendering_suppresses_churn() -> None:
    generator = _load_generator()
    rendered = generator.render_results(
        [
            {
                "model": "Example",
                "config": {"shots": None},
                "metrics": {"accuracy": 0.75},
                "elapsed": 12.345,
                "images": [],
            }
        ],
        stable_metadata=True,
    )

    assert "- Generated: stable" in rendered
    assert "- Git commit: `stable`" in rendered
    assert "not recorded" in rendered
    assert "12.35 s" not in rendered


def test_committed_notebooks_have_outputs_and_validation_blocks_pass_when_present() -> None:
    notebook_paths = sorted((ROOT / "notebooks").glob("*/*.ipynb"))
    assert notebook_paths

    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        outputs = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
        ]
        output_text = "\n".join(
            _notebook_output_text(
                nbformat.from_dict(
                    {
                        "cells": [cell],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    }
                )
            )
            for cell in notebook.cells
            if cell.cell_type == "code"
        )

        assert outputs, f"{path} has no committed outputs"
        if re.search(r"(?im)^validation\b|validation:", output_text):
            assert "passed" in output_text.lower(), f"{path} has validation output without passed"
