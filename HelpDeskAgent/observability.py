"""Explicit Agent Lens spans. Initialize once in the host, never per request."""
import json
import os
from contextlib import contextmanager

from coreweave.forge.agentlens import tracing
from opentelemetry import trace


def initialize(project):
    """Opt in at process startup; keep provider auto-instrumentation disabled."""
    return tracing.init(project, autopatch_integrations=False)


def message(value):
    role = value.get("role", "assistant")
    if role == "assistant":
        return tracing.Message.assistant(
            value.get("content") or "",
            tool_calls=[tracing.ToolCallPart(
                id=tc.get("id", tc["function"]["name"]),
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            ) for tc in value.get("tool_calls") or []],
        )
    if role == "tool":
        return tracing.Message.tool_result(value["tool_call_id"], value["content"])
    return tracing.Message(role=role, content=value.get("content") or "")


@contextmanager
def episode_turn(ticket, agent_name, model_name, user_message, system_prompt):
    # A clarification reply is a NEW user turn in the SAME ticket conversation.
    # Reconstruct only conversation identity across requests, never replay spans.
    with tracing.Conversation(
        conversation_id=ticket.ticket_id, agent_name=agent_name, model=model_name,
        include_content=os.environ.get("AGENT_LENS_INCLUDE_CONTENT", "true").lower() != "false",
        attributes={"helpdesk.category": ticket.category},
    ) as conversation:
        with conversation.start_turn(
            user_message=user_message, system_instructions=[system_prompt],
        ) as turn:
            yield turn


def trace_id():
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def chat(chat_fn, messages, tools_schema, model_name, provider_name):
    turn = tracing.get_current_turn()
    if turn is None:
        return chat_fn(messages, tools_schema)
    with turn.start_llm(model=model_name, provider_name=provider_name) as llm:
        llm.input_messages = [message(m) for m in messages]
        response = chat_fn(messages, tools_schema)
        llm.output_messages = [message({"role": "assistant", **response})]
        llm.response_id = response.get("response_id") or ""
        llm.response_model = response.get("response_model") or ""
        llm.finish_reasons = response.get("finish_reasons") or []
        usage = response.get("usage")
        # OpenAI prompt_tokens already includes cached tokens. Do not add them.
        # Do not turn unavailable usage into zero-valued measured usage.
        if usage is not None:
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens") or 0
            if cached > usage["prompt_tokens"]:
                raise ValueError("cached tokens exceed inclusive prompt tokens")
            llm.usage = tracing.Usage(
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                total_tokens=usage.get("total_tokens"),
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                cache_read_input_tokens=cached,
            )
        return response


def execute_tool(tools, tc):
    name = tc["function"]["name"]
    raw_args = tc["function"].get("arguments") or "{}"
    turn = tracing.get_current_turn()
    # run_episode always owns a turn, even when the SDK is not initialized.
    with turn.start_tool(name=name, arguments=raw_args,
                         tool_call_id=tc.get("id", name)) as span:
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        method = getattr(tools, name, None)
        if method is None:
            result = {"error": f"unknown tool {name}"}
        else:
            try:
                result = method(**args)
            except Exception as exc:
                result = {"error": f"invalid arguments for {name}: {exc}"}
        span.result = json.dumps(result)
        if "error" in result:
            # Preserve the existing recoverable error response AND mark failure.
            span.record_error(RuntimeError(result["error"]))
        return args, result