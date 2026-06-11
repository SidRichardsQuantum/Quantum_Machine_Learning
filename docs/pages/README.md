# Pages tooling

This directory contains the small static-site tooling used by the GitHub Pages workflow.

The generated site is intentionally custom rather than MkDocs-based so it matches the
style of the root portfolio site at:

[https://SidRichardsQuantum.github.io/](https://SidRichardsQuantum.github.io/)

## Files

- `build_site.py` builds `_site/` from the repository Markdown files.
- `generate_results.py` runs deterministic smoke-scale QML workflows and writes
  `docs/results/api-reference.md`.
  It also extracts committed notebook text tables and plots into
  `docs/results/tutorials.md`, `docs/results/real-examples.md`, and
  `docs/results/benchmarks.md`.
- `styles.css` defines the custom portfolio-style visual system for the generated site.
- `assets/reference-results/` stores generated result plots and JSON artifacts embedded by
  `docs/results/api-reference.md`.
- `assets/notebook-results/` stores plots extracted from executed notebooks.

## Generate Results

From the repository root:

```bash
python docs/pages/generate_results.py
```

This refreshes `docs/results/`, the notebook result pages, and generated assets
from the current package APIs and from outputs already stored in notebooks.

The package API result configurations are intentionally small enough to
regenerate locally. Notebook result pages are generated from committed notebook
outputs so the web pages show the same relevant tables and plots a reader sees
in the notebooks. They are reproducible reference outputs, not quantum-advantage
claims.

To refresh notebook-result pages from committed notebook outputs without
rerunning the API reference results:

```bash
python docs/pages/generate_results.py --skip-api-results
```

When notebook outputs need to change, execute the notebooks locally with
Jupyter or `jupyter nbconvert --execute --inplace`, commit the updated
`.ipynb` files, then regenerate and commit the derived result pages and assets.

Useful group execution commands:

```bash
python -m jupyter nbconvert --execute --inplace notebooks/tutorials/*.ipynb
python -m jupyter nbconvert --execute --inplace notebooks/real_examples/*.ipynb
python -m jupyter nbconvert --execute --inplace notebooks/benchmarks/*.ipynb
```

To reduce release-doc churn from generated timestamps, commit hashes, and
runtime values, use stable metadata:

```bash
python docs/pages/generate_results.py --stable-metadata
```

## Build Site

Install the docs build dependencies if needed:

```bash
python -m pip install -e ".[dev]"
```

Then build the static site:

```bash
python docs/pages/build_site.py
```

The generated files are written to `_site/`, which is ignored by git.

## GitHub Pages

The workflow in `.github/workflows/pages.yml`:

1. Checks out the repository.
2. Installs the static site dependencies.
3. Builds `_site/` from committed Markdown and assets.
4. Deploys the Pages artifact.

There is no separate result-refresh workflow. GitHub Pages publishes committed
Markdown and asset files only. Notebook outputs are expected to persist in the
committed `.ipynb` files and are the source of truth for notebook-derived result
pages.
