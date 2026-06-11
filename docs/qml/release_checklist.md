# Release Checklist

Use this checklist before publishing a package release.

## Version and notes

- Bump `pyproject.toml` to the target version.
- Add a top-level `CHANGELOG.md` entry for the same version.
- Update any release-specific documentation that refers to the package version.

## Generated results

- Run the result generator after version bumps or notebook output changes:

```bash
python docs/pages/generate_results.py
```

Use stable metadata for release docs when timestamp, commit, and runtime churn
would obscure meaningful output changes:

```bash
python docs/pages/generate_results.py --stable-metadata
```

- Notebook-derived result pages are extracted from outputs already committed in
  notebooks. When notebook outputs need to change, execute notebooks locally,
  commit the updated `.ipynb` files, then regenerate and commit
  `docs/results/` and `docs/pages/assets/` outputs.
- The result pages embed package version, commit, Python version, and runtime
  metadata, so release version bumps require regenerated result pages.

## Validation

Run the local release checks:

```bash
black .
ruff check . --fix
pytest -q
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
```

Review the final diff and working tree:

```bash
git diff --stat
git status --short
```

## Publish

- Commit the release changes.
- Push and wait for GitHub Actions to pass.
- Tag the exact green release commit:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

- The tag push starts the release workflow chain:
  - `Tests` runs on the tag.
  - `Publish` starts after the matching successful `Tests` run, then publishes
    the checked package artifacts to PyPI for `v*` tags. On `main` pushes, it
    still builds and checks the package without uploading to PyPI.
  - `Pages` deploys after `Publish` completes successfully.
