"""Offline conformance: real SDK serialization; scripted provider ground truth."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from HelpDeskAgent import agent
from HelpDeskAgent import observability as obs


@pytest.fixture(scope="session")
def exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    provider.shutdown()


@pytest.fixture
def spans(exporter, monkeypatch):
    exporter.clear()
    monkeypatch.setattr(obs, "_enabled", True)
    monkeypatch.setattr(obs, "_include_content", True)
    return exporter


def tc(name, arguments=None, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments or {})}}


def reply(*calls, content=None, metadata=None):
    return {"content": content, "tool_calls": list(calls), "_telemetry": metadata}


def row(ticket_id="synthetic-ticket", category="password_reset"):
    return dict(ticket_id=ticket_id, account_id="synthetic-account", category=category,
                text="Reset my password", account_tier="standard", account_device_os="standard",
                account_seats_remaining=10, history_count_30d=0)


def episode(chat_fn, ticket_id="synthetic-ticket", **kwargs):
    world, ticket = agent.world_and_ticket_from_row(row(ticket_id))
    return agent.run_episode(ticket, world, chat_fn, model_name="test-model",
                             provider_name="test-harness", **kwargs)


def by_op(spans, op):
    return [s for s in spans if s.attributes["gen_ai.operation.name"] == op]


def assert_tree(spans, model_count, tool_ids):
    roots = by_op(spans, "invoke_agent")
    assert len(roots) == 1
    root = roots[0]
    assert root.parent is None
    assert len(by_op(spans, "chat")) == model_count
    tools = by_op(spans, "execute_tool")
    assert [s.attributes["gen_ai.tool.call.id"] for s in tools] == tool_ids
    for span in spans:
        assert span.context.trace_id == root.context.trace_id
        assert root.start_time <= span.start_time <= span.end_time <= root.end_time
        if span is not root:
            assert span.parent.span_id == root.context.span_id
    assert not any(k.startswith("gen_ai.usage") for k in root.attributes)
    return root


def test_model_tool_loop_and_delivered_answer(spans):
    events = []
    replies = iter([
        reply(tc("lookup_account", {"account_id": "synthetic-account"}, "lookup"), content="Checking"),
        reply(tc("resolve_ticket", {"kb_article_id": "KB-101", "response_text": "Use MFA"}, "resolve")),
    ])

    def model(messages, tools):
        events.append(json.loads(json.dumps(messages)))
        return next(replies)

    result = episode(model)
    finished = spans.get_finished_spans()
    root = assert_tree(finished, 2, ["lookup", "resolve"])
    assert result["turn_trace_id"] == format(root.context.trace_id, "032x")
    assert root.attributes["helpdesk.outcome"] == "completed"
    chats = by_op(finished, "chat")
    assert json.loads(chats[0].attributes["gen_ai.output.messages"])[0]["parts"] == [
        {"type": "text", "content": "Checking"},
        {"type": "tool_call", "id": "lookup", "name": "lookup_account",
         "arguments": '{"account_id": "synthetic-account"}'},
    ]
    assert events[1][-1]["tool_call_id"] == "lookup"
    assert json.loads(chats[1].attributes["gen_ai.input.messages"])[-1]["parts"][0]["id"] == "lookup"
    assert json.loads(root.attributes["gen_ai.output.messages"])[0]["parts"][0]["content"] == result["response_text"]


def test_clarification_new_submission_same_conversation_no_replay(spans):
    first = agent.ask_agent("I need access to a folder", client="oracle", ticket_id="clarify",
                            category="access_request", hidden_detail="team-drive", score=True)
    assert first["status"] == "pending"
    # The resume payload must remain JSON serializable across HTTP roundtrips.
    resume = json.loads(json.dumps(first["resume"]))
    second = agent.ask_agent("team-drive", client="oracle", resume=resume, score=True)
    assert second["status"] == "done"
    assert first["turn_trace_id"] != second["turn_trace_id"]
    finished = spans.get_finished_spans()
    assert {s.attributes["gen_ai.conversation.id"] for s in finished} == {"helpdesk:clarify"}
    roots = by_op(finished, "invoke_agent")
    assert len(roots) == 2
    assert [s.attributes["gen_ai.tool.name"] for s in by_op(finished, "execute_tool")] == [
        "request_more_info", "resolve_ticket"]
    for root in roots:
        subset = [s for s in finished if s.context.trace_id == root.context.trace_id]
        assert len(by_op(subset, "invoke_agent")) == 1
        assert len(by_op(subset, "chat")) == 1
    assert second["score"]["score"] == 1.0


@pytest.mark.parametrize("same_ticket", [True, False])
def test_identical_submissions_have_distinct_turn_ids(spans, same_ticket):
    first = episode(lambda *_: reply(content="hello"), ticket_id="a")
    second = episode(lambda *_: reply(content="hello"), ticket_id="a" if same_ticket else "b")
    assert first["turn_trace_id"] != second["turn_trace_id"]
    assert len(by_op(spans.get_finished_spans(), "invoke_agent")) == 2


@pytest.mark.parametrize("bad_call", [tc("missing_tool"), tc("resolve_ticket"),
                                      {"id": "broken", "function": {
                                          "name": "lookup_account", "arguments": "not json"}}])
def test_tool_error_then_real_retry_is_preserved(spans, bad_call):
    responses = iter([reply(bad_call), reply(tc("resolve_ticket", {
        "kb_article_id": "KB-101", "response_text": "Resolved"}, "retry"))])
    result = episode(lambda *_: next(responses))
    finished = spans.get_finished_spans()
    assert_tree(finished, 2, [bad_call["id"], "retry"])
    tools = by_op(finished, "execute_tool")
    assert tools[0].status.status_code.name == "ERROR"
    assert "error" in json.loads(tools[0].attributes["gen_ai.tool.call.result"])
    assert tools[1].status.status_code.name != "ERROR"
    assert result["final_action"] is not None


@pytest.mark.parametrize("exception", [RuntimeError("provider unavailable"), KeyboardInterrupt()])
def test_failure_and_cancellation_close_all_spans(spans, exception):
    def fail(*_):
        raise exception

    with pytest.raises(type(exception)):
        episode(fail)
    finished = spans.get_finished_spans()
    assert_tree(finished, 1, [])
    assert all(s.status.status_code.name == "ERROR" for s in finished)
    assert obs._active.get() is None
    assert trace.get_current_span().get_span_context().trace_id == 0


def test_overlap_no_context_leakage(spans):
    barrier = Barrier(2)

    def work(ticket_id):
        def model(*_):
            barrier.wait(timeout=10)
            return reply(tc("resolve_ticket", {"kb_article_id": "KB-101", "response_text": ticket_id}, ticket_id))
        return episode(model, ticket_id=ticket_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(work, ["left", "right"]))
    assert results[0]["turn_trace_id"] != results[1]["turn_trace_id"]
    for ticket_id, result in zip(["left", "right"], results):
        subset = [s for s in spans.get_finished_spans()
                  if format(s.context.trace_id, "032x") == result["turn_trace_id"]]
        assert_tree(subset, 1, [ticket_id])
        assert {s.attributes["gen_ai.conversation.id"] for s in subset} == {f"helpdesk:{ticket_id}"}


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    {"prompt_tokens": 103, "completion_tokens": 7, "total_tokens": 110,
     "prompt_tokens_details": {"cached_tokens": 80},
     "completion_tokens_details": {"reasoning_tokens": 2}}])
def test_usage_only_once_inclusive_cache_missing_not_zero(spans, usage):
    episode(lambda *_: reply(content="answer", metadata={
        "response_id": "response-123", "response_model": "actual-model",
        "finish_reasons": ["stop"], "usage": usage}))
    finished = spans.get_finished_spans()
    assert_tree(finished, 1, [])
    attrs = by_op(finished, "chat")[0].attributes
    assert attrs["gen_ai.response.id"] == "response-123"
    assert attrs["gen_ai.response.model"] == "actual-model"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    if usage:
        assert attrs["gen_ai.usage.input_tokens"] == usage["prompt_tokens"]
        assert attrs["gen_ai.usage.output_tokens"] == usage["completion_tokens"]
        assert attrs["gen_ai.usage.total_tokens"] == usage["total_tokens"]
    else:
        assert not any(k.startswith("gen_ai.usage") for k in attrs)


def test_impossible_cache_retains_evidence_and_flags_invalid(spans):
    episode(lambda *_: reply(metadata={"usage": {
        "prompt_tokens": 2, "prompt_tokens_details": {"cached_tokens": 80}}}))
    attrs = by_op(spans.get_finished_spans(), "chat")[0].attributes
    assert attrs["helpdesk.usage_valid"] is False
    assert attrs["gen_ai.usage.input_tokens"] == 2
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 80


def test_content_suppression_and_fixed_custom_keys(spans, monkeypatch):
    monkeypatch.setattr(obs, "_include_content", False)
    agent.ask_agent("SYNTHETIC_PRIVATE_TICKET", category="password_reset", client="oracle", ticket_id="one")
    keys_one = {k for s in spans.get_finished_spans() for k in s.attributes if k.startswith("helpdesk.")}
    for span in spans.get_finished_spans():
        assert not any(k in span.attributes for k in ["gen_ai.input.messages", "gen_ai.output.messages",
            "gen_ai.tool.call.arguments", "gen_ai.tool.call.result", "gen_ai.system_instructions"])
        assert "SYNTHETIC_PRIVATE_TICKET" not in str(dict(span.attributes))
    spans.clear()
    agent.ask_agent("different ticket", category="hardware_failure", client="oracle", ticket_id="two")
    keys_two = {k for s in spans.get_finished_spans() for k in s.attributes if k.startswith("helpdesk.")}
    assert keys_one == keys_two


def test_disabled_preserves_contract_and_emits_nothing(spans, monkeypatch):
    monkeypatch.setattr(obs, "_enabled", False)
    result = agent.ask_agent("Reset my password", category="password_reset", client="oracle", score=True)
    assert result["turn_trace_id"] is None
    assert result["status"] == "done"
    assert result["score"]["score"] == 1.0
    assert spans.get_finished_spans() == ()


def test_stop_after_first_terminal_no_phantom_dispatch(spans):
    result = episode(lambda *_: reply(
        tc("resolve_ticket", {"kb_article_id": "KB-101", "response_text": "ok"}, "first"),
        tc("missing_tool", call_id="never-executed")))
    assert result["final_action"] is not None
    assert_tree(spans.get_finished_spans(), 1, ["first"])


def test_loop_limit_unresolved_is_visible(spans):
    episode(lambda *_: reply(tc("get_escalation_policy")), max_turns=2)
    root = assert_tree(spans.get_finished_spans(), 2, ["call-1", "call-1"])
    assert root.attributes["helpdesk.outcome"] == "unresolved"


def test_real_openai_adapter_metadata_without_network(spans, monkeypatch):
    completion = SimpleNamespace(
        id="provider-response", model="provider-model", usage=SimpleNamespace(model_dump=lambda **_: {
            "prompt_tokens": 103, "completion_tokens": 7, "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 80}}),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="answer", tool_calls=None))])
    create = Mock(return_value=completion)
    monkeypatch.setattr(agent, "OpenAI", lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result = agent.HelpdeskAgentModel("requested-model", "https://unused.invalid", "synthetic-key").predict(**row())
    assert result["turn_trace_id"]
    create.assert_called_once()
    attrs = by_op(spans.get_finished_spans(), "chat")[0].attributes
    assert attrs["gen_ai.response.id"] == "provider-response"
    assert attrs["gen_ai.response.model"] == "provider-model"
    assert attrs["gen_ai.usage.input_tokens"] == 103


def test_configuration_explicit_idempotent_and_no_autopatching(monkeypatch):
    import weave
    monkeypatch.setattr(obs, "_enabled", False)
    monkeypatch.setattr(obs, "_project", None)
    monkeypatch.setattr(obs, "_include_content", False)
    init = Mock()
    monkeypatch.setattr(weave, "init", init)
    with pytest.raises(ValueError):
        obs.configure_agent_lens("bare-project")
    obs.configure_agent_lens("demo-team/helpdesk")
    obs.configure_agent_lens("demo-team/helpdesk")
    init.assert_called_once()
    settings = init.call_args.kwargs["autopatch_settings"]
    assert all(not getattr(settings, k).enabled for k in type(settings).model_fields)
    with pytest.raises(RuntimeError):
        obs.configure_agent_lens("other-team/project")
    with pytest.raises(RuntimeError):
        obs.configure_agent_lens("demo-team/helpdesk", include_content=True)


def test_known_bad_controls_reject_missing_execution_and_extra_turn(spans):
    episode(lambda *_: reply(tc("get_escalation_policy")), max_turns=1)
    finished = list(spans.get_finished_spans())
    assert_tree(finished, 1, ["call-1"])
    with pytest.raises(AssertionError):
        assert_tree([s for s in finished if s.attributes["gen_ai.operation.name"] != "execute_tool"], 1, ["call-1"])
    with pytest.raises(AssertionError):
        assert_tree(finished + by_op(finished, "invoke_agent"), 1, ["call-1"])


@pytest.mark.parametrize("category", agent.CATEGORIES)
@pytest.mark.parametrize("client", ["oracle", "naive"])
def test_observability_preserves_synthetic_results_across_categories(spans, monkeypatch, category, client):
    kwargs = dict(client=client, category=category, ticket_id="regression",
                  account_tier="executive", account_device_os="unsupported_legacy",
                  account_seats_remaining=0, history_count_30d=4, hidden_detail="finance-reports", score=True)
    observed = agent.ask_agent("Synthetic helpdesk ticket", **kwargs)
    if observed["status"] == "pending":
        observed = agent.ask_agent("finance-reports", client=client, score=True, resume=observed["resume"])
    monkeypatch.setattr(obs, "_enabled", False)
    plain = agent.ask_agent("Synthetic helpdesk ticket", **kwargs)
    if plain["status"] == "pending":
        plain = agent.ask_agent("finance-reports", client=client, score=True, resume=plain["resume"])
    observed["turn_trace_id"] = None
    observed["output"]["turn_trace_id"] = None
    assert observed == plain


@pytest.mark.parametrize("endpoint, provider", [
    ("https://api.openai.com/v1", "openai"),
    ("https://api.inference.wandb.ai/v1", "wandb"),
    ("https://other.example/v1", "openai-compatible"),
])
def test_provider_attribution_uses_endpoint(endpoint, provider):
    model = agent.HelpdeskAgentModel("model", endpoint, "synthetic-key")
    assert model.provider_name == provider


def test_shutdown_flushes_only_at_process_boundary(monkeypatch):
    import weave
    monkeypatch.setattr(obs, "_enabled", True)
    provider = Mock()
    provider.force_flush.return_value = True
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    finish = Mock()
    monkeypatch.setattr(weave, "finish", finish)
    obs.shutdown_agent_lens()
    provider.force_flush.assert_called_once_with()
    finish.assert_called_once_with()
    provider.shutdown.assert_called_once_with()


def test_shutdown_failure_is_not_silently_successful(monkeypatch):
    monkeypatch.setattr(obs, "_enabled", True)
    provider = Mock()
    provider.force_flush.return_value = False
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    with pytest.raises(RuntimeError, match="flush did not complete"):
        obs.shutdown_agent_lens()