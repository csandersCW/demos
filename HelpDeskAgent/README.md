# HelpDeskAgent: Agent Lens observability demo

The helpdesk agent resolves or escalates synthetic IT tickets and can ask the
requester for clarification. Agent Lens instrumentation is **opt-in**: existing
CLI and `ask_agent()` callers work without tracing configuration.

## Setup

Use Python 3.10+ (verified here on Python 3.13):

```bash
python -m pip install -r HelpDeskAgent/requirements-dev.txt
```

This review branch uses the supported **Weave Conversation SDK** ingestion path
for Agent Lens, pinned to `weave==0.53.12`. The newer
`coreweave.forge.agentlens.tracing` SDK package was not available from PyPI during
preparation. There is only one instrumentation path; no provider auto-patching,
legacy `@weave.op` duplication, W&B runs, or W&B artifacts are added.

## Credential-free local demo

From the repository root:

```bash
python -m HelpDeskAgent.agent "My VPN keeps disconnecting" --client oracle --score
python -m HelpDeskAgent.agent "I need access to a folder" \
  --category access_request --client oracle --score
```

The second command asks for a resource. Reply `team-drive` to demonstrate a
second human turn in the same conversation. `oracle` and `naive` are scripted
sanity-check clients, **not model calls**. Their spans, when enabled, are marked
`helpdesk.synthetic=true` and provider `test-harness`; they report no token usage.

## Enable Agent Lens for the demo

Authenticate the **application runtime** with W&B using its normal secret
mechanism (`WANDB_API_KEY`, or a supported local W&B login). Never commit a key.
Choose an existing, authorized `entity/project`; a GitHub repository path is not
a W&B project selection. The application does not infer a destination.

```bash
python -m HelpDeskAgent.agent "My VPN keeps disconnecting" \
  --client oracle --score --agent-lens-project ENTITY/PROJECT --capture-content
```

Replace `ENTITY/PROJECT` with the intended demo destination. Alternatively set
`HELPDESK_AGENT_LENS_PROJECT`. `--capture-content` deliberately opts in to ticket,
model message, and tool payload capture: use synthetic data for demos. Without
it, those payloads are omitted at serialization time. This is **not PII
redaction**: ticket IDs, category, provider metadata, exceptions, and existing
console output remain visible. Do not put secrets in tickets or IDs.

For a real endpoint, pass `--client remote --model MODEL --base-url URL`; the
existing client uses `WANDB_API_KEY`. Use that only with an endpoint accepting
that credential (for example W&B Inference); other providers should supply
their own credential through `ask_agent(api_key=...)`. Tracing authentication
remains separate. This branch does not change model request semantics.

Library/server startup, once before accepting requests:

```python
from HelpDeskAgent.observability import configure_agent_lens, shutdown_agent_lens
from HelpDeskAgent.agent import ask_agent

configure_agent_lens("ENTITY/PROJECT", include_content=True)
try:
    result = ask_agent("Reset my password", client="oracle", ticket_id="demo-unique-001")
    print(result["turn_trace_id"])
finally:
    # Server applications call this from process shutdown, not per request.
    shutdown_agent_lens()
```

Project routing is fixed for the process. Reconfiguration to another destination
or capture policy raises rather than rerouting active spans. Do not initialize
another Weave exporter with a different destination in the same process.

## Trace contract

- `helpdesk:<ticket_id>` is the conversation identity. Callers must supply
  unique IDs for independent tickets and carry the JSON-serializable `resume`
  payload for clarification roundtrips.
- One `invoke_agent` root per **human submission**, not per model iteration.
  Clarification replies are new turns/traces in the same conversation. Completed,
  pending, and unresolved outcomes are distinguished by `helpdesk.outcome`.
- One `chat` span around each actual `chat_fn` invocation, including failures;
  one `execute_tool` span around each dispatch. All are parented to the owning
  turn. Replayed history and calls after the first terminal action do not
  generate execution spans. Real retries are retained.
- Structured assistant text and tool calls/results retain provider call IDs.
  The deterministic customer-facing answer is recorded separately on the turn,
  not inserted into the model's raw output.
- Remote completion ID, actual response model, finish reason, and available
  usage are preserved. OpenAI `prompt_tokens` already includes cached input:
  it is never incremented by cache counts. Missing usage remains absent;
  reported zero remains zero. Usage is never copied onto the parent turn.
  Impossible cache/negative counters remain intact and are flagged with
  `helpdesk.usage_valid=false`, not silently corrected.
- Fixed custom keys: `helpdesk.ticket_id`, `helpdesk.category`,
  `helpdesk.synthetic`, `helpdesk.outcome`, and optional `helpdesk.usage_valid`.
  Ticket IDs and payload fields never become custom attribute **keys**.
- Exceptions and cancellation close spans and propagate unchanged. Recoverable
  tool errors still return to the agent for retry and mark the tool span failed.
  Request-local context is reset even on failure; overlapping requests do not
  share a turn. Flush/shutdown happens at process exit.

## Tests and verification status

```bash
python -m pytest HelpDeskAgent/tests -q
```

The suite uses the **real pinned Conversation SDK**, OTel provider, serializer,
and in-memory exporter with independently scripted responses. No external model,
GitHub, or W&B service is called by the tests. Coverage includes parentage,
multi-step turns, identical submissions, clarification/resume and history replay,
real retries and malformed tool arguments, terminal short-circuiting, loop
exhaustion, provider failure, cancellation, overlapping requests, content
suppression, stable custom keys, missing/zero/cached usage, impossible usage,
the remote OpenAI adapter, and disabled-mode behavior. Known-bad missing-tool
and extra-root controls must fail the tree assertions.

Preparation result: **47 offline tests passed**. An additional comparison with
the unmodified upstream `agent.py` matched all public `ask_agent()` outputs for
**320 combinations** of category, client, account tier, device, seats, and
history, including pending-ticket resumes. CLI oracle smoke testing scored 1.0;
compilation, dependency consistency, and whitespace checks also passed.

Streaming, delegated agents, detached workers, and client-side tool execution
are inapplicable to this synchronous agent. The application owns one server-side
tool dispatcher; no synthetic spans stand in for client execution.

**Pre-demo preparation verifies local emission, not production ingestion or UI
rendering.** No W&B destination was specified, so live traces were not submitted.
Before presenting: run the synthetic oracle example with the actual authorized
destination, query `/agents/spans/query` for the returned `turn_trace_id`, and
compare all returned span IDs, parent IDs, messages, and usage with execution.
Then inspect the same conversation in Agent Lens. Empty/failed queries or a
successful flush alone are not delivery proof. Confirm the follow-up retains
its conversation ID and has a new trace ID. SDK-visible token data is not an
invoice; costs also require matching backend model pricing.
