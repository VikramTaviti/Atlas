# ADR 0001: Initial ATLAS architecture and Milestone 0/1 foundations

- **Status:** Accepted
- **Date:** 2026-11-03
- **Decision owners:** ATLAS maintainers

## Context

ATLAS needs a local-first, testable execution foundation that can accept planner-generated workflows without allowing untrusted proposals to directly execute. The repository was an empty scaffold. The implementation brief specifies Python 3.12+, Pydantic v2, a typed DAG, a custom runtime later, deterministic fakes, and evidence-gated evolution.

## Decisions

1. Use Python 3.12+ and Pydantic v2 for validated, frozen, strict plan contracts.
2. Use JSON Schema Draft 2020-12 validation (`jsonschema`) for declared input/output schemas rather than inventing a partial schema checker.
3. Keep graph compilation independent of any orchestration framework. Use explicit compiler validation and standard-library graph algorithms.
4. Accept agent/tool registries as trusted inputs to the compiler; never allow plan data to define executable code, imports, or registry entries.
5. Canonicalize validated plan JSON with sorted keys and compact separators, then hash using SHA-256. Hashing identifies exact plan content; registry and compiler versions must be separately retained as provenance later.
6. Start with a small argparse CLI to avoid a CLI dependency. Add Typer only if the command surface grows enough to justify it.
7. Keep fake agent/tool adapters synchronous and deterministic as contract fixtures. The future runtime may wrap them in async execution without changing provider-specific details in the IR.
8. Add only pytest, pytest-asyncio, Ruff, and mypy as development tools; no persistence, API, tracing, or provider SDK dependencies before their milestones.

## Alternatives considered

- **LangGraph as core runtime:** rejected for now; would obscure core execution semantics and is explicitly not a hard dependency.
- **Custom schema validation:** rejected because partial JSON Schema support creates surprising acceptance gaps.
- **Typer CLI immediately:** rejected; argparse is sufficient for the initial one-command contract.
- **Plan hash over normalized graph semantics:** rejected; no semantic normalization is defined yet. Hash exact canonical serialized IR and version the compiler separately.

## Consequences

The source is intentionally small and compiler-centric. Python 3.12 is required (including `StrEnum`); the host's available Python 3.9 cannot execute the project. Agent/tool registry versions are not yet embedded in the digest, so persisted provenance must capture them when storage is added. The current concurrency bound is a structural bound on simultaneously ready nodes; the runtime must separately enforce actual worker concurrency.

## Revisit when

The runtime begins supporting branch conditions, worker recovery, external persistence, or parallel registry versioning. Review whether conditional edge semantics need typed expressions, and whether registry identities belong in a compiled artifact envelope.
