# ADR 0004: Centralized deny-by-default tool gateway

## Status

Accepted for Milestone 3.

## Context

Milestone 2 adapters invoked registered tools directly from DAG nodes and the ReAct loop. Registration and an agent profile allowlist alone do not enforce operator capabilities, workspace scope, budgets, approvals, or an auditable policy decision.

## Decision

All tool calls pass through a framework-neutral, per-run `ToolGatewaySession`. Trusted immutable `ToolPolicy` is supplied to the backend by the application; its empty allowlist denies all tools. The gateway validates policy, profile allowlist, capability grants, input/output schemas, byte budgets, workspace containment, timeouts and concurrency, then writes decision and outcome records to a local SQLite append-only API before/after adapter execution. Audit records hold hashes and safe metadata, never raw arguments or outputs. Side-effecting/destructive tools require a fresh action-hash-bound approval; the default approval provider denies. The LangGraph backend only supplies the gateway to tool nodes and ReAct nodes and remains the outer workflow orchestrator.

The first built-ins provide workspace file reading and a test runner that invokes an exact configured argv inside Docker with no network, read-only root/workspace, dropped capabilities, process/memory/CPU limits and bounded time/output. It never invokes a host shell.

## Consequences

- Existing applications must provide `ToolPolicy(allowed_tool_ids=...)` to enable tools.
- Audit persistence is local SQLite for the prototype; multi-worker transactionality, tamper-evidence, retention, and centralized service storage are not claimed.
- The Docker tool requires an operator-provisioned image and Docker daemon; it is intentionally not a general command runner.
- Approval UI/workflow is an adapter seam. No human approval is granted by default.
- Plans and agent outputs remain unable to alter policy.
