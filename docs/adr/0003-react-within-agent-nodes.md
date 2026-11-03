# ADR 0003: Run ReAct inside individual agent nodes

- **Status:** Accepted
- **Date:** 2026-11-03
- **Decision owners:** ATLAS maintainers

## Context

The Milestone 2 executor runs each agent node once. Agents receive no tool-call contract and cannot observe tool results within the same node. ATLAS needs iterative tool use while preserving its task-specific DAG and framework-neutral execution backend.

## Decisions

1. Keep the outer workflow graph unchanged: LangGraph schedules Atlas plan nodes. Add a ReAct runner invoked only from an `agent` node.
2. Extend the agent adapter contract to return either a legacy string final response or a strict structured `AgentTurn`. An `act` turn has an action summary and one or more typed tool calls; a `final` turn has a final output. Do not add a free-form thought/reasoning field.
3. Treat all agent tool calls as untrusted. The runner exposes tools only from the trusted `AgentProfile.allowed_tools` list; it validates arguments and results against trusted `ToolDefinition` JSON Schemas and refuses unregistered or unauthorized calls before invocation.
4. Execute multiple calls within a turn sequentially. Append each success, failure, or rejection to the observation history supplied to the next agent turn.
5. Bound the loop by the profile's `max_iterations` (capped by backend configuration), existing node timeout, configured maximum calls per iteration, and maximum total calls per agent node. Only the structured `final` action terminates successfully; exhausting iterations fails that node.
6. Convert tool errors and rejected calls into concise observations so the agent can recover. Model errors, malformed turns, invalid final outputs, and iteration exhaustion fail the node with safe error summaries.
7. Store structured trace events in LangGraph execution state and `ExecutionResult.agent_traces`: decision/action summary, tool call arguments, observations/results, attempt and iteration numbers, agent identity/version, and explicit node outcome. Do not record private chain-of-thought or raw model responses. Trace/tool-output redaction remains a later security requirement.
8. Preserve the existing async `Agent.invoke` entry point and accept its legacy string return as a one-turn final answer.

## Alternatives considered

- **Make every thought/action a node in the outer LangGraph DAG:** rejected; that would turn a local agent loop into a workflow-topology change and blur the boundary between agent behavior and task orchestration.
- **Use provider-native tool calling as the runtime:** rejected; provider behavior would leak into ATLAS contracts and would not give the backend uniform authorization, validation, limits, or traces.
- **Record complete model reasoning:** rejected; Atlas records action summaries and auditable tool inputs/outputs, not private chain-of-thought.
- **Add a new agent framework or tool-routing dependency:** rejected; the loop uses the existing async protocols, Pydantic, JSON Schema validation, and server-owned registries.

## Consequences

Existing adapters that return strings remain usable for tool-free tasks. Adapters enabling tool use must return `AgentTurn` values (or mappings conforming to that schema) and consume `AgentRequest.observations` on subsequent iterations. Tool availability is an intersection of trusted profile configuration and the backend's trusted registry. The current implementation remains limited to the Milestone 2 risk classes (`read_only` and `idempotent`); it does not add the Phase 3 policy gateway or enable side-effecting tools.

Structured traces intentionally include tool arguments, tool results, and final outputs. Those values may be sensitive; redaction and retention controls remain open work and must precede broader tool integrations.

## Revisit when

Phase 3 adds the policy gateway, or when a provider adapter needs streaming/interruption support. Keep authorization and resource limits in Atlas even if provider-native tool calling is introduced.
