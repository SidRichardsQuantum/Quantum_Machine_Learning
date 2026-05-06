# Pages tooling

This directory contains the small static-site tooling used by the GitHub Pages workflow.

The generated site is intentionally custom rather than MkDocs-based so it matches the
style of the root portfolio site at:

[https://SidRichardsQuantum.github.io/](https://SidRichardsQuantum.github.io/)

## Files

- `build_site.py` builds `_site/` from the repository Markdown files.
- `generate_results.py` runs deterministic smoke-scale QML workflows and writes `RESULTS.md`.
- `styles.css` defines the custom portfolio-style visual system for the generated site.
- `assets/reference-results/` stores generated result plots and JSON artifacts embedded by `RESULTS.md`.

## Generate Results

From the repository root:

```bash
python docs/pages/generate_results.py
```

This refreshes `RESULTS.md` and `docs/pages/assets/reference-results/`.

The result configurations are intentionally small so GitHub Pages can regenerate them in CI.
They are reproducible reference outputs, not quantum-advantage claims.

## Build Site

Install the docs build dependencies if needed:

```bash
python -m pip install markdown pymdown-extensions
```

Then build the static site:

```bash
python docs/pages/build_site.py
```

The generated files are written to `_site/`, which is ignored by git.

## GitHub Pages

The workflow in `.github/workflows/pages.yml`:

1. Checks out the repository.
2. Installs the package and docs dependencies.
3. Regenerates `RESULTS.md` and result images.
4. Builds `_site/`.
5. Deploys the Pages artifact.
