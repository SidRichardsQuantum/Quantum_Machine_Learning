# Result Reports

This directory contains generated result reports used by the custom GitHub Pages site.

- `api-reference.md` contains smoke-scale API reference outputs.
- `tutorials.md` contains tables and plots extracted from tutorial notebooks.
- `real-examples.md` contains tables and plots extracted from real-example notebooks.
- `benchmarks.md` contains tables and plots extracted from benchmark notebooks.

Regenerate these reports from the repository root:

```bash
python docs/pages/generate_results.py
```

Notebook-derived result pages are extracted from outputs already committed in
the notebooks. When notebook outputs need to change, execute notebooks locally,
commit the updated `.ipynb` files, then regenerate and commit the derived
`docs/results/` and `docs/pages/assets/` outputs. GitHub Pages publishes only
committed reports and assets. These reports are reproducible reference outputs,
not quantum-advantage claims.

Generated reports embed package version, git commit, Python version, and runtime
metadata. After a package version bump, rerun the result generator and commit the
updated reports so the release metadata in `docs/results/` matches
`pyproject.toml`.

Use `python docs/pages/generate_results.py --stable-metadata` when release docs
should avoid churn from generated timestamps, commit hashes, and runtime values.
