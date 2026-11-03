# Reproducible fixture report

Regenerate a paired smoke report from a clean checkout:

```bash
atlas compare --suite examples/benchmarks/smoke.json \
  --baseline-id baseline --candidate-id candidate --trials 3 \
  --fixture-only --store /tmp/atlas-evaluation.sqlite \
  > /tmp/atlas-smoke-comparison.json
```

For this versioned fixture suite, the deterministic outcome summary is expected to be:

| Measure | Baseline fixture | Candidate fixture |
|---|---:|---:|
| Paired samples | 6 | 6 |
| Task success rate | 0.50 | 1.00 |
| Policy violations | 0 | 0 |
| Tool invocations | 6 | 3 |

The first case models an omitted verification requirement in the baseline and a passing check in the candidate. These numbers describe only the fixture records in `examples/benchmarks/smoke.json`. They are not empirical model-quality or production-performance claims. Local latency measurements vary by host and should not be compared across machines.
