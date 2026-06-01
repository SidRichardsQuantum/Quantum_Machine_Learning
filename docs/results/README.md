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

Pass `--execute-notebooks` to rerun notebooks before extracting notebook outputs.
These reports are reproducible reference outputs, not quantum-advantage claims.
