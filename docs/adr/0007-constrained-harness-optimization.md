# ADR 0007: Typed harness patches and evidence-gated promotion

## Status

Accepted for Milestone 6.

## Decision

Harness changes are a strict Pydantic patch limited to instruction variant, pre-approved skill IDs, tool preferences already in operator authorization, a verification-step switch, and inference tokens under an operator ceiling. Security policy and other immutable controls are absent from the candidate schema and unknown fields are rejected.

Each candidate is paired against its baseline on the same versioned suite/environment. `IsolatedEvaluationRunner` and `TemporaryWorkspaceProvider` support a fresh copied workspace for each harness/case/trial; command sandboxing remains the runner's responsibility. Promotion eligibility requires zero safety violations, mandatory deterministic checks, uncertainty-aware quality non-inferiority, a predeclared improvement, and cost/latency ceilings. Eligibility is not promotion: a separate approval must match experiment ID and hash and is recorded with the promoted version. Rollback creates a new version from a prior snapshot. Experiment/experience metadata is stored locally in SQLite with scope, evidence IDs, confidence and expiry.

## Consequences

- The optimizer proposes constrained operating-harness changes; it cannot edit permissions or arbitrary topology.
- The local CLI's fixture runner is demonstrative; authenticated operator identity must be supplied by a production approval service.
- Promotion and rollback records are auditable locally but not distributed or host-admin tamper-proof.
