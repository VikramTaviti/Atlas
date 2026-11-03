# Evaluation methodology

## Suite contract

`BenchmarkSuite` is versioned and contains unique `EvaluationCase` IDs, task text, mandatory status, expected output fragments, expected node statuses, expected deterministic test outcomes, and optional fixture metadata. The suite and each completed report are appended to `SQLiteEvaluationStore`; reports record environment ID, harness ID, evaluator version, trial count, per-case outcomes, and aggregate metrics.

Applications provide an `EvaluationRunner` that executes a real workflow with a requested harness ID and returns `EvaluationObservation`. For mutable repository workloads, wrap a `WorkspaceEvaluationRunner` with `IsolatedEvaluationRunner` and `TemporaryWorkspaceProvider` to copy a fresh fixture directory per harness/case/trial. The runner must still enforce command isolation (for example, with the Docker tool) and apply the same environment/budget to both sides; temporary directories alone are not a security sandbox. ATLAS computes deterministic success from the case criteria. A `RubricEvaluator` adapter may add a qualitative score; its evaluator version and model ID are separate fields and do not override deterministic results.

## Metrics

- Task success and deterministic verified quality are reported per case and in aggregate.
- Cost per successful task, total cost, median and p95 latency, retries, tool calls, and policy violations are separate measures.
- Quality variance and an approximate 95% normal interval are reported for at least two samples. With one sample the interval is `[0, 1]` to communicate that uncertainty cannot be estimated.
- Paired comparisons run the same versioned cases, trial indices, and declared environment ID for both harnesses. Candidate improvement and cost deltas are reported without hiding regressions in a single score.
- Any policy violation fails the hard safety gate.

The default fixture suite and `--fixture-only` CLI mode are deterministic educational fixtures. Their timing is measured locally and is not a production performance claim. Production evaluation should use multiple stochastic trials, a versioned environment image, deterministic checks, and separately reported rubric judgments.

## Optimizer promotion gates

The candidate must have zero policy violations, pass every mandatory deterministic case, meet quality non-inferiority using the candidate's uncertainty lower bound, show at least one predeclared improvement (quality, cost, latency, or tool efficiency), and remain within configured cost/latency ceilings. An eligible experiment still requires a separate explicit approval bound to its experiment ID and hash. Promotion is versioned; rollback creates a new version from an earlier snapshot.

The current normal interval uses `1.96 × sample standard error`; it is descriptive and not a substitute for a suitable paired statistical test on larger stochastic benchmarks.
