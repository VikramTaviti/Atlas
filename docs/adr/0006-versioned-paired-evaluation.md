# ADR 0006: Versioned paired benchmark evaluation

## Status

Accepted for Milestone 5.

## Decision

Define versioned suites/cases and a provider-neutral `EvaluationRunner`. Deterministic criteria are evaluated locally. Qualitative scores are optional and separately identify evaluator version and model. Baseline and candidate use identical suite, trial indices, and environment identifier; reports preserve suite snapshots and per-case metadata in a local append-only SQLite store.

Report task success, binary verified quality, sample count/variance/approximate 95% interval, cost, cost per success, median/p95 latency, retries, tool efficiency and safety violations as separate metrics. One-sample intervals use `[0,1]` rather than implying certainty. Safety violations fail a hard gate.

## Consequences

- SQLite is the local prototype store; a production database is an adapter extension.
- `EvaluationRunner` separates benchmark logic from runtime/provider wiring; the CLI's fixture-only runner is explicitly illustrative.
- The normal interval is descriptive, not a claim of robust statistical significance. Larger stochastic benchmarks should use a predeclared paired method and enough trials.
