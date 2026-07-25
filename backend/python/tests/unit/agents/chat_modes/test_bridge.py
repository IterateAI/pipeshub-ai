"""Unit tests for `app.agents.chat_modes.bridge` -- the agent-loop entry
point for ALL THREE `/chat/stream` modes. Mirrors the mocking style of
`tests/unit/agents/adapter/test_sse_bridge.py::TestRunAgentLoopStream`
(same `PipesHubAgentFactory.create`/`AnswerFinalizer.run` patch points,
same fake-agent helpers) since `run_chat_stream` is deliberately the
chat-mode counterpart of `run_agent_loop_stream`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.chat_modes.bridge import (
    _apply_policy_to_chat_state,
    _resolve_web_search_config,
    _with_mode_instructions,
    run_chat_stream,
)
from app.agents.chat_modes.policy import (
    AGENT_POLICY,
    INTERNAL_SEARCH_POLICY,
    WEB_SEARCH_POLICY,
)
from app.agents.chat_modes.prefetch import PrefetchResult
from app.utils.chat_helpers import CitationRefMapper


def _stream_agent(result: Any) -> MagicMock:
    """Same shape `test_sse_bridge.py` uses: `.stream(goal)` yields nothing,
    leaves `.last_stream_result` set to `result`."""
    agent = MagicMock()
    agent.last_stream_result = None

    async def _fake_stream(goal, **kwargs):
        agent.last_stream_result = result
        return
        yield  # pragma: no cover - marks this an async generator

    agent.stream = _fake_stream
    return agent


def _patch_connectors(has_sql: bool = False, has_slack: bool = False):
    return (
        patch(
            "app.utils.execute_query.has_sql_connector_configured",
            new=AsyncMock(return_value=has_sql),
        ),
        patch(
            "app.utils.fetch_slack_thread.has_slack_connector_configured",
            new=AsyncMock(return_value=has_slack),
        ),
    )


class TestModeInstructions:
    def test_known_mode_appends_addendum_to_base_instructions(self) -> None:
        result = _with_mode_instructions("Base prompt.", INTERNAL_SEARCH_POLICY)
        assert result.startswith("Base prompt.")
        assert "internal knowledge base" in result

    def test_known_mode_with_no_base_instructions_returns_addendum_alone(self) -> None:
        result = _with_mode_instructions(None, WEB_SEARCH_POLICY)
        assert "web search" in result.lower()
        assert not result.startswith("\n")

    def test_blank_base_instructions_treated_as_absent(self) -> None:
        result = _with_mode_instructions("   ", AGENT_POLICY)
        assert result == _with_mode_instructions(None, AGENT_POLICY)


class TestApplyPolicyToChatState:
    def test_internal_search_forces_knowledge_true_web_search_off(self) -> None:
        chat_state: dict[str, Any] = {"has_sql_connector": True, "has_slack_connector": True}
        _apply_policy_to_chat_state(chat_state, INTERNAL_SEARCH_POLICY, web_search_config=None)

        assert chat_state["has_knowledge"] is True
        assert chat_state["has_sql_knowledge"] is True
        assert chat_state["has_slack_knowledge"] is True
        assert chat_state["web_search_config"] is None
        assert chat_state["chat_mode"] == "internal_search"

    def test_web_search_disables_knowledge_even_with_connectors_present(self) -> None:
        chat_state: dict[str, Any] = {"has_sql_connector": True, "has_slack_connector": True}
        web_cfg = {"provider": "tavily", "configuration": {}}
        _apply_policy_to_chat_state(chat_state, WEB_SEARCH_POLICY, web_search_config=web_cfg)

        assert chat_state["has_knowledge"] is False
        assert chat_state["has_sql_knowledge"] is False
        assert chat_state["has_slack_knowledge"] is False
        assert chat_state["web_search_config"] == web_cfg

    def test_knowledge_true_but_no_connector_present_keeps_sql_slack_off(self) -> None:
        chat_state: dict[str, Any] = {"has_sql_connector": False, "has_slack_connector": False}
        _apply_policy_to_chat_state(chat_state, AGENT_POLICY, web_search_config=None)

        assert chat_state["has_knowledge"] is True
        assert chat_state["has_sql_knowledge"] is False
        assert chat_state["has_slack_knowledge"] is False


class TestResolveWebSearchConfig:
    async def test_falls_back_to_duckduckgo_when_no_default_provider(self) -> None:
        """Configured providers exist but none is marked default -- the
        Node.js layer treats DuckDuckGo as the active default in that case
        (see `cm_controller.ts::getWebSearchProviders`), so this must not
        disable web_search/fetch_url entirely."""
        config_service = AsyncMock()
        config_service.get_config.return_value = {"providers": [{"provider": "tavily", "isDefault": False}]}

        result = await _resolve_web_search_config(config_service, MagicMock())
        assert result == {"provider": "duckduckgo", "configuration": {}}

    async def test_falls_back_to_duckduckgo_when_no_providers_configured(self) -> None:
        """Brand-new org that has never touched web search settings --
        `providers` is empty/absent. Must still default to DuckDuckGo, not
        silently disable the tool."""
        config_service = AsyncMock()
        config_service.get_config.return_value = {"providers": []}

        result = await _resolve_web_search_config(config_service, MagicMock())
        assert result == {"provider": "duckduckgo", "configuration": {}}

    async def test_returns_default_provider_configuration(self) -> None:
        config_service = AsyncMock()
        config_service.get_config.return_value = {
            "providers": [
                {"provider": "tavily", "isDefault": False},
                {"provider": "serper", "isDefault": True, "configuration": {"apiKey": "x"}},
            ]
        }

        result = await _resolve_web_search_config(config_service, MagicMock())
        assert result == {"provider": "serper", "configuration": {"apiKey": "x"}}

    async def test_config_lookup_failure_returns_none(self) -> None:
        config_service = AsyncMock()
        config_service.get_config.side_effect = RuntimeError("etcd unavailable")

        result = await _resolve_web_search_config(config_service, MagicMock())
        assert result is None


class TestRunChatStream:
    @staticmethod
    def _base_kwargs(policy=None) -> dict[str, Any]:
        config_service = AsyncMock()
        # `config_service.get_config` is itself an `AsyncMock` attribute
        # (child attrs of an `AsyncMock` propagate async-ness), so its
        # return value must be a concrete dict, not the default child
        # mock -- otherwise `_resolve_web_search_config`'s `.get("providers")`
        # silently returns an unawaited coroutine instead of a list.
        config_service.get_config.return_value = {"providers": []}
        return {
            "query_info": {"query": "hello", "chatMode": "agent", "filters": {}},
            "user_info": {"userId": "user-1", "orgId": "org-1"},
            "llm": MagicMock(),
            "policy": policy or AGENT_POLICY,
            "log": MagicMock(),
            "retrieval_service": AsyncMock(),
            "graph_provider": MagicMock(),
            "reranker_service": MagicMock(),
            "config_service": config_service,
        }

    async def test_build_initial_state_failure_yields_error_event(self) -> None:
        sql_patch, slack_patch = _patch_connectors()
        with (
            patch(
                "app.modules.agents.qna.chat_state.build_initial_state",
                side_effect=RuntimeError("boom"),
            ),
            sql_patch,
            slack_patch,
        ):
            events = [chunk async for chunk in run_chat_stream(**self._base_kwargs())]

        assert len(events) == 1
        assert events[0].startswith("event: error\n")
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        assert "boom" not in payload["message"]

    async def test_successful_agent_mode_run_streams_to_completion(self) -> None:
        async def _fake_create(self, context, llm, chat_mode, *, query, model_name=""):
            agent = _stream_agent(MagicMock(success=True, error=None, output="42"))
            return agent, MagicMock(constraints=[]), MagicMock(constraints=[]), []

        async def _fake_finalizer_run(self, *, agent_success, agent_error, event_sink, agent_output=None, streamed_answer="", reasoning_turns=None):
            await event_sink.write({"event": "complete", "data": {"answer": agent_output}})
            return {"answer": agent_output}

        sql_patch, slack_patch = _patch_connectors()
        with (
            patch(
                "app.modules.agents.qna.chat_state.build_initial_state",
                return_value={"org_id": "org-1", "user_id": "user-1", "query": "hello"},
            ),
            sql_patch,
            slack_patch,
            patch("app.agents.chat_modes.bridge.PipesHubAgentFactory.create", new=_fake_create),
            patch("app.agents.chat_modes.bridge.AnswerFinalizer.run", new=_fake_finalizer_run),
        ):
            events = [chunk async for chunk in run_chat_stream(**self._base_kwargs(policy=AGENT_POLICY))]

        assert len(events) == 1
        assert events[0].startswith("event: complete\n")
        assert json.loads(events[0].split("data: ", 1)[1].strip()) == {"answer": "42"}

    async def test_web_search_mode_resolves_provider_config_before_run(self) -> None:
        """`WEB_SEARCH_POLICY.include_web_search=True` must trigger the
        config lookup, and the resolved config must land on `ChatState`
        before `build_initial_state`'s state is handed to the factory."""
        captured_chat_state: dict[str, Any] = {}

        async def _fake_create(self, context, llm, chat_mode, *, query, model_name=""):
            captured_chat_state.update(context.tool_state)
            agent = _stream_agent(MagicMock(success=True, error=None, output="ok"))
            return agent, MagicMock(constraints=[]), MagicMock(constraints=[]), []

        async def _fake_finalizer_run(self, *, agent_success, agent_error, event_sink, agent_output=None, streamed_answer="", reasoning_turns=None):
            await event_sink.write({"event": "complete", "data": {"answer": agent_output}})
            return {"answer": agent_output}

        kwargs = self._base_kwargs(policy=WEB_SEARCH_POLICY)
        kwargs["config_service"].get_config.return_value = {
            "providers": [{"provider": "serper", "isDefault": True, "configuration": {"apiKey": "x"}}]
        }

        sql_patch, slack_patch = _patch_connectors()
        with (
            patch(
                "app.modules.agents.qna.chat_state.build_initial_state",
                return_value={"org_id": "org-1", "user_id": "user-1", "query": "hello"},
            ),
            sql_patch,
            slack_patch,
            patch("app.agents.chat_modes.bridge.PipesHubAgentFactory.create", new=_fake_create),
            patch("app.agents.chat_modes.bridge.AnswerFinalizer.run", new=_fake_finalizer_run),
        ):
            events = [chunk async for chunk in run_chat_stream(**kwargs)]

        kwargs["config_service"].get_config.assert_awaited()
        assert events[-1].startswith("event: complete\n")

    async def test_internal_search_prefetch_merges_into_goal_and_tool_state(self) -> None:
        """`INTERNAL_SEARCH_POLICY.prefetch_retrieval=True` -- a non-empty
        `PrefetchResult` must be folded into `context.tool_state` and
        `goal.constraints`, and `context.internal_search_attempted` set."""
        prefetch_result = PrefetchResult(
            formatted_context="Refunds are processed within 5 business days.",
            final_results=[{"virtual_record_id": "vr1"}],
            virtual_record_id_to_result={"vr1": {"recordId": "r1"}},
            tool_records=[{"recordId": "r1"}],
            citation_ref_mapper=CitationRefMapper(),
            is_empty=False,
        )
        captured: dict[str, Any] = {}

        async def _fake_create(self, context, llm, chat_mode, *, query, model_name=""):
            goal = MagicMock(constraints=[])
            captured["context"] = context
            captured["goal"] = goal
            agent = _stream_agent(MagicMock(success=True, error=None, output="ok"))
            return agent, MagicMock(), goal, []

        async def _fake_finalizer_run(self, *, agent_success, agent_error, event_sink, agent_output=None, streamed_answer="", reasoning_turns=None):
            await event_sink.write({"event": "complete", "data": {"answer": agent_output}})
            return {"answer": agent_output}

        sql_patch, slack_patch = _patch_connectors()
        with (
            patch(
                "app.modules.agents.qna.chat_state.build_initial_state",
                return_value={"org_id": "org-1", "user_id": "user-1", "query": "hello"},
            ),
            sql_patch,
            slack_patch,
            patch("app.agents.chat_modes.bridge.PipesHubAgentFactory.create", new=_fake_create),
            patch("app.agents.chat_modes.bridge.AnswerFinalizer.run", new=_fake_finalizer_run),
            patch(
                "app.agents.chat_modes.bridge.prefetch_retrieval",
                new=AsyncMock(return_value=prefetch_result),
            ),
        ):
            events = [
                chunk
                async for chunk in run_chat_stream(**self._base_kwargs(policy=INTERNAL_SEARCH_POLICY))
            ]

        assert events[-1].startswith("event: complete\n")
        assert captured["context"].internal_search_attempted is True
        assert captured["context"].tool_state["final_results"] == prefetch_result.final_results
        assert any(
            "Refunds are processed" in str(c) for c in captured["goal"].constraints
        )

    async def test_agent_run_failure_emits_error_event(self) -> None:
        async def _fake_create(self, context, llm, chat_mode, *, query, model_name=""):
            raise RuntimeError("transport exploded")

        sql_patch, slack_patch = _patch_connectors()
        with (
            patch(
                "app.modules.agents.qna.chat_state.build_initial_state",
                return_value={"org_id": "org-1", "user_id": "user-1", "query": "hello"},
            ),
            sql_patch,
            slack_patch,
            patch("app.agents.chat_modes.bridge.PipesHubAgentFactory.create", new=_fake_create),
        ):
            events = [chunk async for chunk in run_chat_stream(**self._base_kwargs())]

        assert len(events) == 1
        assert events[0].startswith("event: error\n")
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        assert "transport exploded" not in payload["message"]

    async def test_sandbox_manager_destroyed_on_completion(self) -> None:
        sandbox_manager = MagicMock()
        sandbox_manager.destroy_all = AsyncMock()

        async def _fake_create(self, context, llm, chat_mode, *, query, model_name=""):
            context.sandbox_manager = sandbox_manager
            agent = _stream_agent(MagicMock(success=True, error=None, output="ok"))
            return agent, MagicMock(), MagicMock(constraints=[]), []

        async def _fake_finalizer_run(self, *, agent_success, agent_error, event_sink, agent_output=None, streamed_answer="", reasoning_turns=None):
            await event_sink.write({"event": "complete", "data": {"answer": "ok"}})
            return {"answer": "ok"}

        sql_patch, slack_patch = _patch_connectors()
        with (
            patch(
                "app.modules.agents.qna.chat_state.build_initial_state",
                return_value={"org_id": "org-1", "user_id": "user-1", "query": "hello"},
            ),
            sql_patch,
            slack_patch,
            patch("app.agents.chat_modes.bridge.PipesHubAgentFactory.create", new=_fake_create),
            patch("app.agents.chat_modes.bridge.AnswerFinalizer.run", new=_fake_finalizer_run),
        ):
            events = [chunk async for chunk in run_chat_stream(**self._base_kwargs())]

        assert events[-1].startswith("event: complete\n")
        sandbox_manager.destroy_all.assert_awaited_once()


    async def test_client_disconnect_mid_stream_cancels_producer_and_running_agent(self) -> None:
        """Mirrors `run_agent_loop_stream()`'s disconnect-cancellation
        contract (see `test_sse_bridge.py`'s "detached_and_orphaned_spawn_
        tasks" regression tests): when the SSE consumer stops pulling
        (client disconnect -> FastAPI cancels the task awaiting `body_
        iterator.__anext__()`), `run_chat_stream()`'s `finally` block must
        cancel the still-running `_produce()` task rather than let the
        agent keep executing against an abandoned request."""
        agent_stream_started = asyncio.Event()
        agent_stream_cancelled = asyncio.Event()

        async def _hanging_stream(goal, **kwargs):
            agent_stream_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                agent_stream_cancelled.set()
                raise
            return
            yield  # pragma: no cover - marks this an async generator

        agent = MagicMock()
        agent.last_stream_result = None
        agent.stream = _hanging_stream

        async def _fake_create(self, context, llm, chat_mode, *, query, model_name=""):
            return agent, MagicMock(constraints=[]), MagicMock(constraints=[]), []

        sql_patch, slack_patch = _patch_connectors()
        with (
            patch(
                "app.modules.agents.qna.chat_state.build_initial_state",
                return_value={"org_id": "org-1", "user_id": "user-1", "query": "hello"},
            ),
            sql_patch,
            slack_patch,
            patch("app.agents.chat_modes.bridge.PipesHubAgentFactory.create", new=_fake_create),
        ):
            agen = run_chat_stream(**self._base_kwargs())
            pull_task = asyncio.ensure_future(agen.__anext__())
            await asyncio.wait_for(agent_stream_started.wait(), timeout=2)

            # Simulate the client going away: cancel the task that was
            # awaiting the next SSE frame, exactly as Starlette does when
            # the connection drops mid-response.
            pull_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pull_task

            await asyncio.wait_for(agent_stream_cancelled.wait(), timeout=2)
            await agen.aclose()


class TestRunChatStreamNoToolsDegradation:
    """`supports_tool_calls=False` (Ollama) path."""

    @staticmethod
    def _base_kwargs(policy) -> dict[str, Any]:
        return {
            "query_info": {"query": "hello", "chatMode": policy.name, "filters": {}},
            "user_info": {"userId": "user-1", "orgId": "org-1"},
            "llm": MagicMock(),
            "policy": policy,
            "log": MagicMock(),
            "retrieval_service": AsyncMock(),
            "graph_provider": MagicMock(),
            "reranker_service": MagicMock(),
            "config_service": AsyncMock(),
            "supports_tool_calls": False,
        }

    async def test_web_search_mode_fails_fast_with_unsupported_model_error(self) -> None:
        events = [chunk async for chunk in run_chat_stream(**self._base_kwargs(WEB_SEARCH_POLICY))]

        assert len(events) == 1
        assert events[0].startswith("event: error\n")
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        assert payload["type"] == "unsupported_model_mode"

    async def test_agent_mode_fails_fast_with_unsupported_model_error(self) -> None:
        events = [chunk async for chunk in run_chat_stream(**self._base_kwargs(AGENT_POLICY))]

        assert len(events) == 1
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        assert payload["type"] == "unsupported_model_mode"

    async def test_internal_search_mode_degrades_to_forced_prefetch_and_streams(self) -> None:
        prefetch_result = PrefetchResult(
            formatted_context="Context for no-tools answer.",
            final_results=[{"virtual_record_id": "vr1"}],
            virtual_record_id_to_result={},
            tool_records=[],
            citation_ref_mapper=CitationRefMapper(),
            is_empty=False,
        )

        async def _fake_stream_llm_response_with_tools(**kwargs):
            yield {"event": "answer_chunk", "data": {"chunk": "Answer", "accumulated": "Answer"}}
            yield {"event": "complete", "data": {"answer": "Answer"}}

        with (
            patch(
                "app.agents.chat_modes.bridge.prefetch_retrieval",
                new=AsyncMock(return_value=prefetch_result),
            ) as mock_prefetch,
            patch(
                "app.agents.chat_modes.bridge.stream_llm_response_with_tools",
                new=_fake_stream_llm_response_with_tools,
            ),
        ):
            events = [
                chunk
                async for chunk in run_chat_stream(**self._base_kwargs(INTERNAL_SEARCH_POLICY))
            ]

        # `force=True` regardless of follow-up detection -- no tool exists to fall back on.
        assert mock_prefetch.call_args.kwargs["force"] is True
        event_names = [chunk.split("\n", 1)[0] for chunk in events]
        assert "event: status" in event_names
        assert event_names[-1] == "event: complete"
