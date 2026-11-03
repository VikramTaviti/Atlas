# ADR 0008: Standard-library local dashboard and explicit fixture CLI

## Status

Accepted for Milestone 7.

## Decision

Keep the CLI on stdlib `argparse` and implement a minimal read-only HTTP dashboard with `http.server`, avoiding a web framework dependency for a local read-only view. Fixture evaluation/optimization are available only with an explicit `--fixture-only` flag. Docker Compose runs the credential-free dynamic DAG example and persists runtime databases/artifacts to a bind mount. CI runs the full static/test suite; tag pushes build package artifacts and create a GitHub release.

## Consequences

- The dashboard has no authentication and is intended for loopback/local use.
- Real workflow execution/evaluation is application-wired through current runtime and runner interfaces; no hosted service or provider SDK is installed.
- Video capture is not automated by repository tooling; a reproducible demo script and command sequence are provided for recording.
