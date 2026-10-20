"""Opt-in Agent Lens spans using the supported Weave Conversation SDK.

No initialization, network calls, or provider patching occurs on import.
Initialize once at process startup, before accepting concurrent requests.
"""

from contextvars import ContextVar
from functools import wraps
from inspect import signature

_enabled = False
_project = None
_include_content = False
_active = ContextVar("helpdesk_agent_lens", default=None)


def configure_agent_lens(project, *, include_content=False):
    """Enable tracing for an explicit entity/project; never reroute active work."""
    global _enabled, _project, _include_content
    if (len(project.split("/")) != 2 or not all(project.split("/"))
            or project != project.strip() or any(c.isspace() for c in project)):
        raise ValueError("Agent Lens requires an explicit entity/project")
    if _enabled:
        if (_project, _include_content) != (project, include_content):
            raise RuntimeError("Agent Lens is already configured; restart to change routing")
        return
    import weave
    from weave.trace.autopatch import AutopatchSettings

    # Explicit spans own all model/tool capture, so disable every auto-patcher.
    weave.init(project, autopatch_settings=AutopatchSettings(
        **{key: {"enabled": False} for key in AutopatchSettings.model_fields}
    ))
    _project, _include_content, _enabled = project, include_content, True


def shutdown_agent_lens():
    """Flush at process exit, never at a request boundary."""
    if _enabled:
        import weave
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush") and not provider.force_flush():
            raise RuntimeError("Agent Lens flush did not complete")
        weave.finish()
        if hasattr(provider, "shutdown"):
            provider.shutdown()


def _message(message):
    from weave.conversation import Message, ToolCallPart

    role = message["role"]
    if role == "assistant":
        return Message.assistant(message.get("content") or "", tool_calls=[
            ToolCallPart(id=tc.get("id", tc["function"]["name"]),
                         name=tc["function"]["name"],
                         arguments=tc["function"].get("arguments") or "{}")
            for tc in message.get("tool_calls") or []
        ])
    if role == "tool":
        return Message.tool_result(message["tool_call_id"], message["content"])
    return Message(role=role, content=message.get("content") or "")


def observe_episode(function):
    """One human submission -> one turn, including every internal model step."""
    sig = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        if not _enabled:
            return function(*args, **kwargs)
        from opentelemetry import trace
        from weave.conversation import Conversation, Message

        params = sig.bind(*args, **kwargs)
        params.apply_defaults()
        p = params.arguments
        ticket = p["ticket"]
        user_message = (p["follow_up_message"] if p["prior_messages"] is not None
                        else f"New ticket from account {ticket.account_id}:\n{ticket.text}")
        with Conversation(
            conversation_id=f"helpdesk:{ticket.ticket_id}",
            agent_name=p["agent_name"], model=p["model_name"],
            include_content=_include_content,
            attributes={"helpdesk.ticket_id": ticket.ticket_id,
                        "helpdesk.category": ticket.category,
                        "helpdesk.synthetic": p["provider_name"] == "test-harness"},
        ) as conversation:
            with conversation.start_turn(user_message=user_message) as turn:
                trace_id = format(trace.get_current_span().get_span_context().trace_id, "032x")
                token = _active.set((turn, p["model_name"], p["provider_name"]))
                try:
                    result = function(*args, **kwargs)
                    # The delivered customer response is not invented model text.
                    turn.record(output_messages=[Message.assistant(result["response_text"])])
                    turn.set_attributes({"helpdesk.outcome": (
                        "pending" if result["status"] == "pending" else
                        "completed" if result["final_action"] is not None else "unresolved"
                    )})
                    result["turn_trace_id"] = trace_id if int(trace_id, 16) else None
                    return result
                finally:
                    _active.reset(token)

    return wrapped


def call_model(chat_fn, messages, tools_schema):
    active = _active.get()
    if active is None:
        return chat_fn(messages, tools_schema)
    turn, model, provider = active
    with turn.start_llm(model=model, provider_name=provider) as llm:
        llm.input_messages = [_message(m) for m in messages]
        response = chat_fn(messages, tools_schema)
        llm.output_messages = [_message({"role": "assistant", **response})]
        metadata = response.get("_telemetry") or {}
        llm.record(response_id=metadata.get("response_id"),
                   response_model=metadata.get("response_model"),
                   finish_reasons=metadata.get("finish_reasons"))
        usage = metadata.get("usage") or {}
        # OpenAI prompt_tokens already includes cached tokens: never add them.
        counters = {"gen_ai.usage.input_tokens": usage.get("prompt_tokens"),
                    "gen_ai.usage.output_tokens": usage.get("completion_tokens"),
                    "gen_ai.usage.total_tokens": usage.get("total_tokens"),
                    "gen_ai.usage.cache_read.input_tokens":
                        (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "gen_ai.usage.reasoning_tokens":
                        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        counters = {k: v for k, v in counters.items() if v is not None}
        cached = counters.get("gen_ai.usage.cache_read.input_tokens", 0)
        total = counters.get("gen_ai.usage.input_tokens")
        if any(v < 0 for v in counters.values()) or (total is not None and cached > total):
            llm.set_attributes({"helpdesk.usage_valid": False})
        elif counters:
            llm.set_attributes({"helpdesk.usage_valid": True})
        # Retain impossible counters as evidence instead of silently fixing them.
        if counters:
            llm.set_attributes(counters)
        return response


def call_tool(dispatch, *, name, arguments, tool_call_id):
    active = _active.get()
    if active is None:
        return dispatch()
    turn, _, _ = active
    with turn.start_tool(name=name, arguments=arguments, tool_call_id=tool_call_id) as tool:
        result = dispatch()
        tool.result = result
        if "error" in result:
            # Detailed result obeys content capture; error metadata stays generic.
            tool.record_error(RuntimeError("Helpdesk tool execution failed"))
        return result