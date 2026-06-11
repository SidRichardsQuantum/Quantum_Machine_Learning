# Pages tooling

This directory contains the small static-site tooling used by the GitHub Pages workflow.

The generated site is intentionally custom rather than MkDocs-based so it matches the
style of the root portfolio site at:

[https://SidRichardsQuantum.github.io/](https://SidRichardsQuantum.github.io/)

## Files

- `build_site.py` builds `_site/` from the repository Markdown files.
- `generate_results.py` runs deterministic smoke-scale QML workflows and writes
  `docs/results/api-reference.md`.
  It can also execute notebooks and extract their printed tables and plots into
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

This refreshes `docs/results/`, the notebook result pages, and generated assets.

The package API result configurations are intentionally small enough to
regenerate in CI. Notebook result pages are generated from executed notebooks so
the web pages show the same relevant tables and plots a reader sees in the
notebooks. They are reproducible reference outputs, not quantum-advantage
claims.

To refresh notebook-result pages from already executed notebooks without rerunning
the notebooks:

```bash
python docs/pages/generate_results.py --skip-api-results
```

To execute notebooks first:

```bash
python docs/pages/generate_results.py --skip-api-results --execute-notebooks
```

To execute only selected notebooks before regenerating notebook result pages:

```bash
python docs/pages/generate_results.py --skip-api-results \
  --execute-notebook notebooks/tutorials/01-classical-vs-quantum-classifier.ipynb
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

Notebook execution checks are handled separately by
`.github/workflows/refresh-results.yml`. That workflow runs for pull requests
and `main` pushes that touch notebooks, QML source, the result generator,
dependencies, or the workflow itself. Notebook-only changes execute just the
changed notebooks before regenerating notebook result pages from committed
notebook outputs. Source, generator, dependency, or manual runs execute the full
notebook set and regenerate all result artifacts.

The workflow restores executed notebooks, then verifies that `docs/results/` and
`docs/pages/assets/` are unchanged. It does not commit or push to `main`. If the
check fails, regenerate results locally, commit the updated generated artifacts,
and push that commit for review.
