# ADR 0010: Gemini adapter on the existing Agent/ReAct boundary

## Status

Accepted for Milestone 9.

## Context

Atlas already defines a provider-independent `Agent` protocol, typed `AgentRequest` and
`AgentTurn` contracts, and a ReAct runner that owns authorization/tool execution. A real
provider should add model capability without moving tools outside the policy gateway or
making Google SDK types part of the core runtime.

## Decision

- Implement `GeminiAgent` as an optional adapter backed by Google's official
  `google-genai` Python SDK and its async client.
- Keep provider SDK imports lazy; `FakeAgent` and core use remain independent of the
  Gemini extra.
- Map Gemini manual function-call requests to Atlas `AgentToolCall`s. Supply declaration
  schemas only, disable SDK automatic function execution, and let the existing ReAct
  runner call the gateway and send observations in the next request.
- Parse final decisions into the existing `AgentTurn` contract. Keep provider/model,
  measured latency, usage counts, and optional operator-priced cost in generic
  `AgentTelemetry` rather than adding Google-specific runtime contracts.
- Read credentials from `GOOGLE_API_KEY`/`GEMINI_API_KEY` or a protected file path.
  Never log raw SDK error details or key material.
- Use mocked provider tests in standard CI and gate any real API smoke test behind an
  explicit opt-in environment flag.

## Consequences

- Google is the first provider, but OpenAI and Anthropic can implement the same Agent
  protocol without changes to PlanCompiler or LangGraph.
- Inference usage/cost is per decision when the provider returns token counts. Cost is
  an estimate only when rates are configured; the provider does not define the user's
  billing terms.
- The Gemini adapter is not automatically installed into the default API registry;
  trusted application code must register it with a server-owned `AgentProfile` and
  gateway policy.
- The deterministic FakeAgent remains the default for normal tests and local fixture
  examples. Live calls can consume quota and are never made by standard CI.

## Alternatives considered

- **Provider-specific branching in ReAct/LangGraph:** rejected because it couples the
  runtime to one SDK and complicates future providers.
- **SDK-managed automatic function execution:** rejected because it bypasses Atlas's
  policy gateway.
- **Make the Gemini SDK a core dependency:** rejected to preserve fast, credential-free
  framework-neutral use.
