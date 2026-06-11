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

Pass `--execute-notebooks` to rerun all notebooks before extracting notebook
outputs, or pass `--execute-notebook <path>` one or more times to rerun only
selected notebooks. GitHub Pages publishes the committed reports and assets; the
`Refresh results` workflow executes notebooks and verifies that generated
artifacts are current when relevant notebook, QML source, result-generation, or
dependency files change. If the check fails, regenerate the reports locally and
commit the updated `docs/results/` and `docs/pages/assets/` outputs. These
reports are reproducible reference outputs, not quantum-advantage claims.
