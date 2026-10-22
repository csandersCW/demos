# HelpDeskAgent — Agent Lens observability demo

## Setup

Requires Python 3.10+. From this directory:

```bash
python -m pip install -r requirements.txt
```

Configure `WANDB_API_KEY` through your runtime's secret mechanism. Never commit
keys or `.env` files. Set `AGENT_LENS_PROJECT=wandb-pmm/HelpDeskAgent` to opt in
to export, or pass `--trace-project wandb-pmm/HelpDeskAgent` to the CLI.
Use the entity/project you are authorized to write to if running elsewhere.

## Deterministic demo (no inference cost)

```bash
python agent.py 'My VPN keeps disconnecting' --client oracle --score \
  --trace-project wandb-pmm/HelpDeskAgent
python agent.py 'I need access to a folder' --client oracle \
  --category access_request --trace-project wandb-pmm/HelpDeskAgent
```

The second example asks a question; reply `team-drive`. View the ticket in
[Agent Lens Conversations](https://forge.coreweave.com/agent-lens/wandb-pmm/HelpDeskAgent).
The oracle/naive clients are **synthetic test harnesses**, not real model calls;
their spans are labeled `oracle-dummy` / `naive-dummy` and `test-harness`.
They intentionally have no measured token usage or inference cost.

For real inference use `--client remote --model` and `--base-url` for your
configured OpenAI-compatible endpoint. The existing default remains OpenAI;
explicitly pass the intended endpoint and its credentials rather than sending
a W&B key to an unrelated service. Real-model demo validation is separate from
the deterministic instrumentation tests.

## Hosting in an application

```python
from HelpDeskAgent import observability
from HelpDeskAgent.agent import ask_agent

observability.initialize("wandb-pmm/HelpDeskAgent")  # once at process startup
result = ask_agent("My VPN keeps disconnecting", client="oracle")
# On clarification: ask_agent(reply, client="oracle", resume=result["resume"])
# At process exit (after requests finish): observability.tracing.shutdown()
```

Library callers must initialize explicitly; importing the module or calling
`ask_agent` does not enable network export. The CLI reads `AGENT_LENS_PROJECT`.
Do not reinitialize project routing, flush, or shut down per web request.
The SDK also registers process-exit shutdown. Provider autopatching is disabled
because the application creates its own explicit model spans.

## What is captured

- A conversation is identified by ticket ID (retain it across clarification
  requests; supply unique ticket IDs for independent tickets).
- Each user submission owns one `invoke_agent` turn through the entire model/
  tool loop. A clarification reply starts a distinct turn in the same conversation.
- Each model request is a child `chat` span with exact message/tool-call parts,
  response ID, actual model, finish reason, and usage when the provider supplies it.
  OpenAI prompt totals already include cached tokens; cache counts are not added.
- Each actual dispatch is a child `execute_tool` span with original call ID,
  arguments, result, duration, and error status, including recoverable failures.
  History replay is never exported as execution. The terminal/pending stop rule
  remains unchanged; extra unexecuted tool calls are not exported as executions.
- The turn records the delivered console/customer summary separately from raw
  model output, and `turn_trace_id` is returned to callers (None when disabled).
- Fixed custom key `helpdesk.category` supports filtering without dynamic keys.

Content capture is enabled for this synthetic demo. Set
`AGENT_LENS_INCLUDE_CONTENT=false` to omit message/tool payloads. This is **not
PII redaction**: custom attributes and exception text still require review.
This demo has synchronous non-streaming model calls, server-side tools, and no
delegated workers; streaming/client-tool/delegation coverage is not claimed.

## Tests

```bash
python -m pip install 'pytest>=8,<10'
python -m pytest -q
```

Tests use the actual SDK with an in-memory exporter and stubbed provider
responses: no inference calls, backend writes, W&B runs, or artifacts. They check
turn/child parentage, clarification identity, history exclusion, exact call IDs,
usage, terminal-stop behavior, provider metadata, errors, cancellation, and
concurrent requests. Local tests do not establish backend delivery or UI rendering;
verify the intended deployment's Agent Lens view before presenting a live demo.

### Pre-demo verification (2026-10-22)

- 16 offline tests passed against `coreweave==0.1.0b0` on Python 3.13.
- A synthetic oracle ticket was exported to `wandb-pmm/HelpDeskAgent` and read
  back through `/agents/spans/query`: 7 spans (1 `invoke_agent`, 3 `chat`,
  3 `execute_tool`), matching parentage, tool-call IDs, and delivered output.
  Trace ID: `37b2b36e8d89c08e6761d68f0a4a364e`.
- Real-model inference and browser rendering were not exercised. The tests
  stub provider responses; the backend smoke uses the deterministic oracle.
- No PR, W&B run, or W&B artifact is part of this preparation.
