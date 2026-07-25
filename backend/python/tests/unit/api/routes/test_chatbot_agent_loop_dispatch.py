"""Coverage for `askAIStream`'s agent-loop dispatch (`chatbot.py`'s
`_use_agent_loop_for_chat`/`_generate_chat_stream_via_agent_loop`) -- the
default `/chat/stream` path since the LangGraph removal migration.

`tests/unit/api/routes/test_chatbot*.py` cover the pre-migration direct-LLM
pipeline (forced on via the `_force_legacy_chat_pipeline` autouse fixture in
those files); `tests/unit/agents/chat_modes/test_bridge.py` covers
`run_chat_stream()` itself in depth. This file is the seam between the
route layer and that package: dispatch decisions and the `ChatQuery` ->
plain-dict adaptation `_generate_chat_stream_via_agent_loop()` performs.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.chat_modes.policy import AGENT_POLICY, WEB_SEARCH_POLICY


class TestUseAgentLoopForChat:
    def test_defaults_to_true_when_unset(self, monkeypatch):
        from app.api.routes.chatbot import _use_agent_loop_for_chat

        monkeypatch.delenv("PIPESHUB_CHAT_USE_AGENT_LOOP", raising=False)
        assert _use_agent_loop_for_chat() is True

    def test_explicit_false_disables_it(self, monkeypatch):
        from app.api.routes.chatbot import _use_agent_loop_for_chat

        monkeypatch.setenv("PIPESHUB_CHAT_USE_AGENT_LOOP", "false")
        assert _use_agent_loop_for_chat() is False

    @pytest.mark.parametrize("value", ["True", "TRUE", "  true  ".strip()])
    def test_case_and_whitespace_insensitive_true(self, monkeypatch, value):
        from app.api.routes.chatbot import _use_agent_loop_for_chat

        monkeypatch.setenv("PIPESHUB_CHAT_USE_AGENT_LOOP", value)
        assert _use_agent_loop_for_chat() is True

    def test_unrecognized_value_is_falsy(self, monkeypatch):
        from app.api.routes.chatbot import _use_agent_loop_for_chat

        monkeypatch.setenv("PIPESHUB_CHAT_USE_AGENT_LOOP", "nope")
        assert _use_agent_loop_for_chat() is False


class TestGenerateChatStreamViaAgentLoop:
    @staticmethod
    def _mock_request(body: dict) -> MagicMock:
        request = MagicMock()
        request.state.user = {"orgId": "org-1", "userId": "user-1", "email": "u@corp.com"}
        request.query_params = {"sendUserInfo": True}
        request.app.container.logger.return_value = MagicMock()
        return request

    async def test_llm_init_failure_yields_error_without_calling_run_chat_stream(self):
        from app.api.routes.chatbot import ChatQuery, _generate_chat_stream_via_agent_loop

        request = self._mock_request({})
        query_info = ChatQuery(query="hello")

        with (
            patch("app.api.routes.chatbot.get_llm_for_chat", new=AsyncMock(return_value=None)),
            patch("app.api.routes.chatbot.run_chat_stream") as mock_run_chat_stream,
        ):
            events = [
                chunk
                async for chunk in _generate_chat_stream_via_agent_loop(
                    request, query_info, AsyncMock(), MagicMock(), AsyncMock(),
                )
            ]

        mock_run_chat_stream.assert_not_called()
        assert len(events) == 1
        assert "error" in events[0]

    async def test_llm_init_failure_with_agui_protocol_yields_run_error(self):
        from app.api.routes.chatbot import ChatQuery, _generate_chat_stream_via_agent_loop

        request = self._mock_request({})
        query_info = ChatQuery(query="hello", protocol="agui")

        with (
            patch("app.api.routes.chatbot.get_llm_for_chat", new=AsyncMock(return_value=None)),
            patch("app.api.routes.chatbot.run_chat_stream") as mock_run_chat_stream,
        ):
            events = [
                chunk
                async for chunk in _generate_chat_stream_via_agent_loop(
                    request, query_info, AsyncMock(), MagicMock(), AsyncMock(),
                )
            ]

        mock_run_chat_stream.assert_not_called()
        assert len(events) == 1
        assert events[0].startswith("event: RUN_ERROR\n")
        assert '"type": "RUN_ERROR"' in events[0]

    async def test_protocol_defaults_to_legacy_and_forwards_agui_when_requested(self):
        from app.api.routes.chatbot import ChatQuery, _generate_chat_stream_via_agent_loop

        request = self._mock_request({})
        query_info = ChatQuery(query="hello", protocol="agui")

        async def _fake_run_chat_stream(*args, **kwargs):
            yield "event: RUN_FINISHED\ndata: {}\n\n"

        with (
            patch(
                "app.api.routes.chatbot.get_llm_for_chat",
                new=AsyncMock(return_value=(MagicMock(), {"provider": "openai", "isMultimodal": False}, {})),
            ),
            patch(
                "app.api.routes.chatbot.run_chat_stream", side_effect=_fake_run_chat_stream,
            ) as mock_run_chat_stream,
        ):
            [
                chunk
                async for chunk in _generate_chat_stream_via_agent_loop(
                    request, query_info, AsyncMock(), MagicMock(), AsyncMock(),
                )
            ]

        assert mock_run_chat_stream.call_args.kwargs["protocol"] == "agui"

    async def test_builds_query_dict_and_user_info_and_forwards_policy(self):
        """The plain-dict contract `run_chat_stream()` expects -- see
        `chat_modes/bridge.py`'s module docstring for why it never sees
        `ChatQuery`/`Request` directly."""
        from app.api.routes.chatbot import ChatQuery, _generate_chat_stream_via_agent_loop

        request = self._mock_request({})
        query_info = ChatQuery(
            query="what's our refund policy?",
            chatMode="agent",
            conversationId="conv-1",
            attachments=[{"virtualRecordId": "vr1", "mimeType": "application/pdf"}],
            filters={"apps": ["confluence"]},
        )

        async def _fake_run_chat_stream(*args, **kwargs):
            yield "event: complete\ndata: {}\n\n"

        with (
            patch(
                "app.api.routes.chatbot.get_llm_for_chat",
                new=AsyncMock(return_value=(MagicMock(), {"provider": "openai", "isMultimodal": False}, {})),
            ),
            patch(
                "app.api.routes.chatbot.run_chat_stream", side_effect=_fake_run_chat_stream,
            ) as mock_run_chat_stream,
        ):
            events = [
                chunk
                async for chunk in _generate_chat_stream_via_agent_loop(
                    request, query_info, AsyncMock(), MagicMock(), AsyncMock(),
                )
            ]

        assert events == ["event: complete\ndata: {}\n\n"]
        mock_run_chat_stream.assert_called_once()
        call_args = mock_run_chat_stream.call_args
        query_dict, user_info = call_args.args[0], call_args.args[1]

        assert query_dict["query"] == "what's our refund policy?"
        assert query_dict["chatMode"] == "agent"
        assert query_dict["conversationId"] == "conv-1"
        assert query_dict["attachments"] == [{"virtualRecordId": "vr1", "mimeType": "application/pdf"}]
        assert query_dict["filters"] == {"apps": ["confluence"]}

        assert user_info["userId"] == "user-1"
        assert user_info["orgId"] == "org-1"
        assert user_info["userEmail"] == "u@corp.com"

        policy_arg = call_args.args[3]
        assert policy_arg is AGENT_POLICY
        assert call_args.kwargs["protocol"] == "legacy"

    async def test_ollama_provider_keeps_tool_calls_enabled(self):
        """Ollama must not be forced into the no-tools degradation path --
        `internal_search` needs tool calling to reach `fetch_full_record`
        mid-run (see `_generate_chat_stream_via_agent_loop`'s docstring).
        `run_chat_stream`'s `supports_tool_calls` defaults to `True`, so this
        asserts the chat route doesn't override that default for Ollama."""
        from app.api.routes.chatbot import ChatQuery, _generate_chat_stream_via_agent_loop

        request = self._mock_request({})
        query_info = ChatQuery(query="hello", chatMode="web_search")

        async def _fake_run_chat_stream(*args, **kwargs):
            yield "event: status\ndata: {}\n\n"

        with (
            patch(
                "app.api.routes.chatbot.get_llm_for_chat",
                new=AsyncMock(return_value=(MagicMock(), {"provider": "ollama", "isMultimodal": False}, {})),
            ),
            patch(
                "app.api.routes.chatbot.run_chat_stream", side_effect=_fake_run_chat_stream,
            ) as mock_run_chat_stream,
        ):
            [
                chunk
                async for chunk in _generate_chat_stream_via_agent_loop(
                    request, query_info, AsyncMock(), MagicMock(), AsyncMock(),
                )
            ]

        assert "supports_tool_calls" not in mock_run_chat_stream.call_args.kwargs
        assert mock_run_chat_stream.call_args.args[3] is WEB_SEARCH_POLICY


class TestAskAIStreamDefaultDispatch:
    """Default (`PIPESHUB_CHAT_USE_AGENT_LOOP` unset) -- every `chatMode`
    routes through the agent loop, unlike the pre-migration dispatch which
    branched on `chatMode == "web_search"` between two direct-LLM helpers."""

    @staticmethod
    def _mock_request(body: dict) -> MagicMock:
        request = MagicMock()
        request.state.user = {"orgId": "org-1", "userId": "user-1", "email": "u@corp.com"}
        request.query_params = {"sendUserInfo": True}
        request.json = AsyncMock(return_value=body)
        request.app.container.logger.return_value = MagicMock()
        return request

    @pytest.mark.parametrize("chat_mode", ["agent", "internal_search", "web_search"])
    async def test_every_chat_mode_dispatches_to_agent_loop(self, monkeypatch, chat_mode):
        from app.api.routes.chatbot import askAIStream

        monkeypatch.delenv("PIPESHUB_CHAT_USE_AGENT_LOOP", raising=False)
        request = self._mock_request({"query": "hello", "chatMode": chat_mode})

        async def _fake_agent_loop_stream(**kwargs):
            yield "event: complete\ndata: {}\n\n"

        with (
            patch(
                "app.api.routes.chatbot._generate_chat_stream_via_agent_loop",
                side_effect=_fake_agent_loop_stream,
            ) as mock_agent_loop,
            patch("app.api.routes.chatbot._generate_web_search_stream") as mock_web_search,
            patch("app.api.routes.chatbot._generate_internal_search_stream") as mock_internal_search,
        ):
            response = await askAIStream(
                request=request, retrieval_service=AsyncMock(), graph_provider=AsyncMock(),
                config_service=AsyncMock(),
            )
            events = [chunk async for chunk in response.body_iterator]

        assert events == ["event: complete\ndata: {}\n\n"]
        mock_agent_loop.assert_called_once()
        mock_web_search.assert_not_called()
        mock_internal_search.assert_not_called()

    async def test_explicit_false_falls_back_to_legacy_web_search_dispatch(self, monkeypatch):
        from app.api.routes.chatbot import askAIStream

        monkeypatch.setenv("PIPESHUB_CHAT_USE_AGENT_LOOP", "false")
        request = self._mock_request({"query": "hello", "chatMode": "web_search"})

        async def _fake_web_search_stream(**kwargs):
            yield "event: complete\ndata: {}\n\n"

        with (
            patch("app.api.routes.chatbot._generate_chat_stream_via_agent_loop") as mock_agent_loop,
            patch(
                "app.api.routes.chatbot._generate_web_search_stream",
                side_effect=_fake_web_search_stream,
            ) as mock_web_search,
        ):
            response = await askAIStream(
                request=request, retrieval_service=AsyncMock(), graph_provider=AsyncMock(),
                config_service=AsyncMock(),
            )
            events = [chunk async for chunk in response.body_iterator]

        assert events == ["event: complete\ndata: {}\n\n"]
        mock_web_search.assert_called_once()
        mock_agent_loop.assert_not_called()

    async def test_explicit_false_falls_back_to_legacy_internal_search_dispatch(self, monkeypatch):
        from app.api.routes.chatbot import askAIStream

        monkeypatch.setenv("PIPESHUB_CHAT_USE_AGENT_LOOP", "false")
        request = self._mock_request({"query": "hello"})

        async def _fake_internal_search_stream(**kwargs):
            yield "event: complete\ndata: {}\n\n"

        with (
            patch("app.api.routes.chatbot._generate_chat_stream_via_agent_loop") as mock_agent_loop,
            patch(
                "app.api.routes.chatbot._generate_internal_search_stream",
                side_effect=_fake_internal_search_stream,
            ) as mock_internal_search,
        ):
            response = await askAIStream(
                request=request, retrieval_service=AsyncMock(), graph_provider=AsyncMock(),
                config_service=AsyncMock(),
            )
            events = [chunk async for chunk in response.body_iterator]

        assert events == ["event: complete\ndata: {}\n\n"]
        mock_internal_search.assert_called_once()
        mock_agent_loop.assert_not_called()
