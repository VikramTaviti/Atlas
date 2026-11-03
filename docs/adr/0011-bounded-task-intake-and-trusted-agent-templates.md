# ADR 0011: Bounded task intake and trusted agent templates

## Status

Accepted

## Context

Atlas needs a first integrated path from natural-language task to dynamic multi-agent graph without allowing the model to define runtime capabilities. Existing Plan IR, compiler, provider-independent Agent contract, ReAct runner, LangGraph adapter, gateway, and asynchronous API are already established seams.

## Decision

- Define a provider-independent `Planner` protocol returning a closed `WorkflowProposal` containing only node IDs, trusted role IDs, bounded task descriptions, dependency edges, depth, and parallelism.
- Implement `GeminiPlanner` using the existing Gemini SDK/configuration and structured JSON output. The planner has no tool-execution interface.
- Resolve role IDs through an application-owned `AgentTemplateRegistry`. Template profiles own allowed tools/capabilities; template factories create one fresh adapter per graph node.
- Compile task-generated Plan IR through `PlanCompiler` before run admission. Execute it through an ephemeral LangGraph backend using the same configured tool registry and `ToolPolicy` gateway.
- Add `POST /v1/tasks` separately from `POST /v1/runs`. Task planning is enabled only when trusted planner and template dependencies are injected.
- Close ephemeral adapters in a `finally` block and close the planner when the API application shuts down.

## Consequences

Task intake reuses existing execution/security contracts and keeps generated plans reviewable in the acceptance response. The configured API process still owns single-instance run admission and local persistence. The live example requires Gemini credentials and Docker. Result aggregation remains the ordinary output node envelope, and evaluation, diagnosis, optimization, and promotion are not part of this slice.

## Security boundaries

The planner cannot specify tools, permissions, capabilities, credentials, provider settings, or policy fields. Unknown roles and extra fields fail closed. Template creation rejects unregistered allowed tools. The runtime checks each ReAct call through the existing deny-by-default gateway. Operator limits are supplied server-side and checked again before compilation.
