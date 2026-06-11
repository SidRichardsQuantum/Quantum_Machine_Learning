# Notebook Authoring Guide

This guide defines the expected structure for tutorial, benchmark, and
real-example notebooks in this repository. Notebooks should remain thin clients
over reusable package APIs in `src/qml/`; reusable algorithms, estimators,
metrics, and benchmark helpers belong in the package first.

## Shared Requirements

Every notebook should:

- run from the repository root, `notebooks/`, and its own directory
- locate the repository root before importing `qml`
- use deterministic seeds and record them in visible configuration
- keep default sample counts and optimizer steps small enough for docs refreshes
- import QML workflows from `src/qml/` instead of reimplementing them inline
- use `qml.reporting.format_table`, `print_table`, or `print_section` for
  compact text output
- produce only relevant plots and keep them readable at documentation size
- avoid quantum-advantage language unless backed by a dedicated study
- end with a validation block that includes a boolean `passed` field

Use ASCII table output for result blocks. The result generator extracts printed
tables and PNG plots from executed notebooks into `docs/results/`.

## Repository Root Setup

Use this pattern near the top of notebooks:

```python
from pathlib import Path
import sys

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
```

Keep imports grouped after this block. Prefer package imports over relative
notebook utilities.

## Tutorial Notebooks

Tutorial notebooks live in `notebooks/tutorials/`.

They should explain one public API or closely related API family. Include:

- a short title and scope statement
- one deterministic configuration block
- one minimal package-backed run
- a compact table showing key inputs and metrics
- one or two plots when they clarify behavior
- a validation block with `passed`

Tutorials should prioritize API clarity over benchmark strength. Avoid broad
model rankings and avoid expensive sweeps.

## Benchmark Notebooks

Benchmark notebooks live in `notebooks/benchmarks/`.

They should answer one narrow comparison question. Include:

- QML and classical model lists
- dataset, split, noise, seed, shot, and model-specific settings
- multi-seed summaries when feasible
- train/test metrics and generalization gaps
- runtime summaries
- paired deltas against the best included classical baseline where applicable
- plots that show metric/runtime tradeoffs, intervals, scaling, shots, noise, or
  capacity
- wording that defaults are smoke-scale templates
- a validation block with `passed`

Use package benchmark helpers such as `compare_classification_models(...)`,
`compare_regression_models(...)`, and `benchmark_runtime_scaling(...)` when the
notebook fits those result structures.

## Real-Example Notebooks

Real-example notebooks live in `notebooks/real_examples/`.

They may contain deterministic domain simulators or feature construction, but
the QML model should come from `src/qml/`. Include:

- a clearly named domain problem
- compact deterministic data generation
- feature and target definitions
- at least one classical baseline
- package-backed QML model usage
- domain-relevant metrics
- succinct plots and tables
- interpretation that states whether the baseline or QML model performed better
- a validation block with `passed`

Prefer fewer high-quality domain examples over many shallow examples.

## Validation Blocks

The final executable cell should print a compact validation table. Include
enough fields to verify the notebook ran the intended path:

```python
validation = {
    "samples": len(x),
    "features": x.shape[1],
    "primary_metric": metric_value,
    "baseline_metric": baseline_value,
    "passed": bool(condition),
}

print(format_table(validation, title="Validation"))
```

The `passed` condition should check correctness or sanity, not just that the
cell executed. Examples include nonempty result rows, finite metrics,
minimum accuracy, maximum error, or expected model output shapes.

## Plot Capture

The result generator captures `image/png` outputs from executed notebooks. In
most notebooks, `plt.show()` is enough. When running with the non-interactive
`Agg` backend and a plot is not captured, display an explicit PNG:

```python
from io import BytesIO
from IPython.display import Image, display

buffer = BytesIO()
fig.savefig(buffer, format="png", dpi=150)
display(Image(data=buffer.getvalue()))
plt.close(fig)
```

Use clear axis labels and legends. Avoid oversized figures, dense annotations,
and plots that duplicate table content without adding interpretation.

## Result Page Generation

Notebook outputs are committed and are the source of truth for notebook-derived
result pages. Regenerate result pages from the repository root after committing
or preparing updated notebook outputs:

```bash
python docs/pages/generate_results.py --skip-api-results
```

When notebook outputs need to change, execute the notebooks locally with
Jupyter or `jupyter nbconvert --execute --inplace`, commit the updated
`.ipynb` files, then regenerate and commit the derived `docs/results/` and
`docs/pages/assets/` outputs. There is no separate CI result-refresh workflow;
GitHub Pages publishes committed Markdown and asset files only.

Useful group execution commands:

```bash
python -m jupyter nbconvert --execute --inplace notebooks/tutorials/*.ipynb
python -m jupyter nbconvert --execute --inplace notebooks/real_examples/*.ipynb
python -m jupyter nbconvert --execute --inplace notebooks/benchmarks/*.ipynb
```

For release docs where timestamp, commit, and runtime churn is not useful, pass
stable metadata when regenerating result pages:

```bash
python docs/pages/generate_results.py --stable-metadata
```

## Review Checklist

Before adding or updating a notebook, verify:

- the notebook imports reusable package APIs
- defaults are small and deterministic
- classical baselines are included for benchmarks and real examples
- result tables use shared reporting helpers
- plots are captured in `docs/results/`
- validation includes `passed`
- notebook outputs are committed when they change
- generated result pages and assets are regenerated from committed notebook outputs
- `python docs/pages/build_site.py` succeeds
