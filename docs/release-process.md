# Release process

1. Run the documented Python 3.12 setup, `ruff check .`, `mypy src/atlas`, and `pytest` locally.
2. Verify the working tree and README commands from a clean checkout. Do not include `.env`, credentials, local databases, or artifact payloads.
3. Update the package version in `pyproject.toml`, release notes, and `ATLAS_CONTEXT.md`.
4. Push a signed/versioned tag matching `v*`. GitHub Actions reruns the package build and publishes the wheel/source distribution as GitHub Release assets with generated notes.
5. Review the release artifacts and limitations before announcing them. No automatic PyPI publication is configured.

The tagged workflow requires repository Actions to permit the `contents: write` token permission.
