"""Real SDK serializer/exporter tests; no model calls or backend writes."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import agent
from coreweave.forge.agentlens import tracing
from coreweave.forge.agentlens.tracing import tracing as sdk


@pytest.fixture
def exported(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(sdk, "OTLPSpanExporter", lambda **kw: exporter)
    agent.observability.initialize("wandb-pmm/HelpDeskAgent")
    yield exporter
    tracing.shutdown()


def spans(exported):
    tracing.force_flush()
    return list(exported.get_finished_spans())


def op(span):
    return span.attributes["gen_ai.operation.name"]


def check_tree(items, turns=1):
    roots = [s for s in items if op(s) == "invoke_agent"]
    assert len(roots) == turns
    assert len({s.context.trace_id for s in roots}) == turns
    by_trace = {s.context.trace_id: s for s in roots}
    for s in items:
        if op(s) != "invoke_agent":
            assert s.parent.span_id == by_trace[s.context.trace_id].context.span_id
            assert s.start_time >= by_trace[s.context.trace_id].start_time
            assert s.end_time <= by_trace[s.context.trace_id].end_time
    assert all("gen_ai.usage.input_tokens" not in s.attributes for s in roots)
    return roots


def tc(name, args, call_id="original-call"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}


def episode(chat_fn, prior_messages=None, prior_tool_calls=None):
    world, ticket = agent.world_and_ticket_from_row(agent._build_row(
        "synthetic-ticket", "My password is broken", "password_reset",
        "standard", "standard", 10, 0,
    ))
    return agent.run_episode(ticket, world, chat_fn, prior_messages=prior_messages,
                             prior_tool_calls=prior_tool_calls, follow_up_message="same reply")


def test_multistep_usage_and_delivery(exported):
    calls = []
    def fake(messages, schema):
        calls.append(list(messages))
        response = {"content": None, "tool_calls": [
            tc("lookup_account", {"account_id": "cli-account"}, "lookup")
            if len(calls) == 1 else tc("resolve_ticket", {
                "kb_article_id": "KB-001", "response_text": "Reset complete"}, "finish")],
            "response_id": f"response-{len(calls)}", "response_model": "synthetic-model",
            "finish_reasons": ["tool_calls"],
            "usage": {"prompt_tokens": 103, "completion_tokens": 7, "total_tokens": 110,
                      "prompt_tokens_details": {"cached_tokens": 80}}}
        return response
    output = episode(fake)
    items = spans(exported)
    roots = check_tree(items)
    assert len(calls) == 2
    assert output["turn_trace_id"] == format(roots[0].context.trace_id, "032x")
    chats = [s for s in items if op(s) == "chat"]
    tools = [s for s in items if op(s) == "execute_tool"]
    assert len(chats) == len(tools) == 2
    assert [s.attributes["gen_ai.tool.call.id"] for s in tools] == ["lookup", "finish"]
    for s in chats:
        assert s.attributes["gen_ai.usage.input_tokens"] == 103
        assert s.attributes["gen_ai.usage.output_tokens"] == 7
        assert "synthetic-model" == s.attributes["gen_ai.response.model"]
    assert "Reset complete" in json.dumps(output["messages"])
    assert "tool_call" in chats[0].attributes["gen_ai.output.messages"]
    assert "Resolution:" in roots[0].attributes["gen_ai.output.messages"]


def test_clarification_resume_no_history_execution(exported):
    first = agent.ask_agent("I need access to a folder", client="oracle", ticket_id="conversation-1",
                            category="access_request", hidden_detail="team-drive")
    assert first["status"] == "pending"
    second = agent.ask_agent("team-drive", client="oracle", resume=first["resume"])
    assert second["status"] == "done"
    items = spans(exported)
    roots = check_tree(items, turns=2)
    assert first["turn_trace_id"] != second["turn_trace_id"]
    assert all(s.attributes["gen_ai.conversation.id"] == "conversation-1" for s in items)
    tools = [s for s in items if op(s) == "execute_tool"]
    assert len(tools) == 2  # history is context, not another execution
    assert len(roots) == 2


@pytest.mark.parametrize("failure", [RuntimeError("provider unavailable"), KeyboardInterrupt()])
def test_failure_and_cancellation_close_spans(exported, failure):
    def fail(*args):
        raise failure
    with pytest.raises(type(failure)):
        episode(fail)
    items = spans(exported)
    check_tree(items)
    assert len(items) == 2
    assert all(s.status.status_code == StatusCode.ERROR for s in items)


@pytest.mark.parametrize("name,args", [("missing_tool", {}), ("resolve_ticket", {}),
                                       ("lookup_account", {"account_id": "absent"})])
def test_tool_errors_preserved_and_retry(exported, name, args):
    count = 0
    def fake(messages, schema):
        nonlocal count
        count += 1
        return {"tool_calls": [tc(name, args)]} if count == 1 else {"content": "No action"}
    output = episode(fake)
    items = spans(exported)
    check_tree(items)
    tool = next(s for s in items if op(s) == "execute_tool")
    assert tool.status.status_code == StatusCode.ERROR
    assert "error" in output["messages"][3]["content"]
    assert len([s for s in items if op(s) == "chat"]) == 2


def test_overlapping_identical_submissions(exported):
    barrier = Barrier(2)
    def fake(messages, schema):
        barrier.wait(timeout=5)
        return {"content": "same answer"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: episode(fake), range(2)))
    assert results[0]["turn_trace_id"] != results[1]["turn_trace_id"]
    check_tree(spans(exported), turns=2)


def test_missing_usage_is_not_measured_zero(exported):
    episode(lambda *args: {"content": "No counters"})
    chat = next(s for s in spans(exported) if op(s) == "chat")
    assert "gen_ai.usage.input_tokens" not in chat.attributes
    assert "gen_ai.usage.output_tokens" not in chat.attributes


def test_impossible_cached_usage_rejected(exported):
    with pytest.raises(ValueError, match="inclusive"):
        episode(lambda *args: {"content": "bad counters", "usage": {
            "prompt_tokens": 3, "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 80}}})
    assert all(s.status.status_code == StatusCode.ERROR for s in spans(exported))


def test_terminal_stops_extra_dispatch(exported):
    episode(lambda *args: {"tool_calls": [
        tc("resolve_ticket", {"kb_article_id": "KB-001", "response_text": "done"}),
        tc("missing_tool", {}, "must-not-execute")]})
    assert len([s for s in spans(exported) if op(s) == "execute_tool"]) == 1


def test_provider_metadata_extraction(exported, monkeypatch):
    response = SimpleNamespace(
        id="provider-id", model="actual-model", usage=SimpleNamespace(model_dump=lambda: {
            "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content="actual text", tool_calls=None))])
    monkeypatch.setattr(agent, "OpenAI", lambda **kw: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: response))))
    row = agent._build_row("metadata", "password", "password_reset", "standard", "standard", 10, 0)
    agent.HelpdeskAgentModel("requested-model", "https://unused.invalid", "unused").predict(**row)
    chat = next(s for s in spans(exported) if op(s) == "chat")
    assert chat.attributes["gen_ai.response.id"] == "provider-id"
    assert chat.attributes["gen_ai.response.model"] == "actual-model"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 10


def test_content_suppression(exported, monkeypatch):
    monkeypatch.setenv("AGENT_LENS_INCLUDE_CONTENT", "false")
    episode(lambda *args: {"tool_calls": [tc("resolve_ticket", {
        "kb_article_id": "KB-001", "response_text": "private payload"})]})
    for span in spans(exported):
        assert "gen_ai.input.messages" not in span.attributes
        assert "gen_ai.output.messages" not in span.attributes
        assert "gen_ai.tool.call.arguments" not in span.attributes
        assert "gen_ai.tool.call.result" not in span.attributes
        assert "private payload" not in json.dumps(dict(span.attributes))


def test_repeated_real_attempts_not_deduplicated(exported):
    count = 0
    def fake(*args):
        nonlocal count
        count += 1
        return {"tool_calls": [tc("missing_tool", {}, f"attempt-{count}")]} if count < 3 else {"content": "done"}
    episode(fake)
    tools = [s for s in spans(exported) if op(s) == "execute_tool"]
    assert [s.attributes["gen_ai.tool.call.id"] for s in tools] == ["attempt-1", "attempt-2"]
    assert all(s.status.status_code == StatusCode.ERROR for s in tools)


def test_malformed_json_still_reports_error(exported):
    count = 0
    def fake(*args):
        nonlocal count
        count += 1
        call = tc("resolve_ticket", {})
        call["function"]["arguments"] = "{broken"
        return {"tool_calls": [call]} if count == 1 else {"content": "done"}
    output = episode(fake)
    tool = next(s for s in spans(exported) if op(s) == "execute_tool")
    assert tool.status.status_code == StatusCode.ERROR
    assert "invalid arguments" in output["messages"][3]["content"]


def test_conformance_rejects_known_bad_controls(exported):
    episode(lambda *args: {"content": "done"})
    items = spans(exported)
    check_tree(items)
    # A duplicated per-step root must fail the one-submission/one-turn check.
    with pytest.raises(AssertionError):
        check_tree(items + [next(s for s in items if op(s) == "invoke_agent")])
    # A detached child must fail parentage checks.
    child = next(s for s in items if op(s) == "chat")
    bad = SimpleNamespace(context=child.context, attributes=child.attributes,
                          parent=SimpleNamespace(span_id=0))
    with pytest.raises(AssertionError):
        check_tree([next(s for s in items if op(s) == "invoke_agent"), bad])