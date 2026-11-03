# Provider adapters

## Provider-independent contract

Adapters implement `Agent.invoke(AgentRequest)` from `atlas.agents` and return an
`AgentTurn` (or the legacy final string). `AgentRequest` carries the instruction,
serialized run/dependency/mapped input, iteration number, accumulated tool observations,
and the authorized `AgentToolSpec` list. The agent returns requested calls as typed
`AgentToolCall`s; the adapter must never execute tool code. Atlas's ReAct runner applies
limits and authorization and sends every tool request through the policy gateway.

`AgentTurn.telemetry` is provider-neutral. When available it records provider/model ID,
request latency, prompt/completion token usage, and optional estimated cost. Cost is only
estimated when both token usage and operator-configured rates are present.

## Gemini adapter

Milestone 9 adds `GeminiAgent`, implemented with Google's official `google-genai`
Python SDK. Install it with:

```bash
python -m pip install -e '.[gemini,langgraph]'
```

The adapter uses the SDK's async client, manual function declarations, and typed
`AgentTurn` parsing. It explicitly disables automatic SDK function execution. Gemini
function requests are translated to Atlas tool IDs and returned to the ReAct loop. The
next model request is rebuilt as Gemini conversation turns: each prior model function
call is followed by a matching user function response, including safe outputs or failure
summaries. Atlas retains the SDK's original model content in the in-memory ReAct turn so
Gemini function call IDs and thought signatures are passed back unchanged; the response
echoes the corresponding call ID. This opaque continuation is not written to traces.
Fallback reconstruction uses Atlas observations, call IDs, and iteration grouping. The
declared tool schema is rebuilt from the authorized tool specs for each request. Provider
errors retain the SDK's numeric code, status string, and a bounded, secret-redacted message
as structured trace fields and in the safe error summary. The system instruction asks Gemini to return a final turn when
observed results answer the task; Atlas still enforces its configured iteration cap.
Nonempty plain-text final responses are mapped to `AgentTurn(action="final", ...)`; JSON
responses must validate against the strict `AgentTurn` schema. Empty text, malformed JSON,
and JSON missing required fields fail with sanitized Pydantic diagnostics containing field,
error type, and expected value, without echoing the provider's input values. If a valid
`AgentTurn` carries a final value that violates its agent output schema, Atlas reports the
JSON Schema field, failed constraint, expected type, and actual value type without including
the value itself.

The key may be supplied as `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or a file path in
`GEMINI_API_KEY_FILE` (the container default path is `/run/secrets/gemini_api_key`).
Explicit constructor arguments are also available for deployment wiring and tests.
Environment key values take precedence over the file. Do not put credentials into Plan
IR, agent instructions, event attributes, traces, or source control. The adapter
preserves SDK HTTP code/status and a bounded redacted provider message in diagnostics. It
does not copy raw exception text or unsanitized credentials into logs or traces.

Generation settings are configurable through `GeminiAgentConfig` or these optional
environment variables: `ATLAS_GEMINI_MODEL` (default `gemini-3.5-flash-lite`; falls back to
`ATLAS_DEFAULT_MODEL` when set),
`ATLAS_GEMINI_TEMPERATURE`, `ATLAS_GEMINI_TIMEOUT_SECONDS`, and
`ATLAS_GEMINI_MAX_OUTPUT_TOKENS`. Optional
`ATLAS_GEMINI_INPUT_PRICE_PER_MILLION_USD` and
`ATLAS_GEMINI_OUTPUT_PRICE_PER_MILLION_USD` values support an estimated cost field.
Rates are deployment-owned and model/account-specific; estimates are not billing data.
Free-tier quota and model availability depend on Google's current account/model terms.
Google currently limits Gemini 2.5 model access to existing users; new projects should
use a currently available model such as `gemini-3.5-flash-lite` or `gemini-3.8-flash`.

### Local smoke run

Set the key in the process environment without adding it to a command argument, or use
the ignored local `.env` with Docker Compose. For local shell execution, load a filled
`.env` as exported variables:

```bash
cp -n .env.example .env  # only creates .env if it does not already exist
# Add the key to .env using a local editor, then:
set -a
. ./.env
set +a
python examples/gemini_react.py
```

The example registers one local catalog tool in trusted application code and explicitly
authorizes it with `ToolPolicy`. Gemini can only request that tool; Atlas executes it
through its gateway. The output trace includes decision summaries and safe telemetry,
not private chain-of-thought or credentials.

For Compose environment injection, the default `docker-compose.yml` passes configured
`GEMINI_API_KEY`/`GOOGLE_API_KEY` and model settings into the container. For a mounted
Compose secret, place the key in a file ignored under `secrets/`, set
`ATLAS_GEMINI_KEY_FILE` to its path, and start with the optional override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gemini.yml up --build -d
```

Keep `GEMINI_API_KEY` and `GOOGLE_API_KEY` unset/blank when using the file secret, so the
adapter reads `/run/secrets/gemini_api_key`.

An opt-in live smoke test is kept out of normal CI. With credentials configured, run:

```bash
ATLAS_GEMINI_SMOKE=1 python -m pytest -m integration -q tests/test_gemini_smoke.py
```

Mocked SDK tests run in the standard suite and make no network calls.

## Future providers

OpenAI and Anthropic adapters can implement the same `Agent` protocol and return
provider-neutral `AgentTurn`/`AgentTelemetry` values. They should follow the same
boundary: provider function-call suggestions become Atlas `AgentToolCall`s; only Atlas
executes tools through the policy gateway. No OpenAI or Anthropic adapter is included
in this milestone.

For real provider evaluations, an application-owned `EvaluationRunner` should invoke
the compiled workflow and return measured observations. Keep `RubricEvaluator`
separate, version the evaluator, and retain qualitative scores apart from deterministic
acceptance checks. `FakeAgent` remains the deterministic default for tests and
credential-free examples.
