"""Tests for opt-in sandbox tools during response synthesis."""

import logging
from unittest.mock import MagicMock, patch

from app.modules.agents.qna.nodes import _get_response_execution_tools


def named_tool(name: str, original_name: str | None = None) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool._original_name = original_name or name
    return tool


def test_response_execution_tools_disabled_by_default() -> None:
    with patch.dict("os.environ", {}, clear=True), patch(
        "app.modules.agents.qna.tool_system.get_agent_tools_with_schemas"
    ) as load_tools:
        assert _get_response_execution_tools({}, logging.getLogger(__name__)) == []
        load_tools.assert_not_called()


def test_response_execution_tools_include_only_sandbox_tools() -> None:
    tools = [
        named_tool("coding_sandbox_execute_python", "coding_sandbox.execute_python"),
        named_tool("database_sandbox_execute_sqlite", "database_sandbox.execute_sqlite"),
        named_tool("retrieval_search_internal_knowledge"),
        named_tool("internaltools_ask_user_question"),
    ]
    with patch.dict(
        "os.environ", {"RESPOND_NODE_EXECUTION_TOOLS_ENABLED": "true"}, clear=True
    ), patch(
        "app.modules.agents.qna.tool_system.get_agent_tools_with_schemas",
        return_value=tools,
    ):
        selected = _get_response_execution_tools({}, logging.getLogger(__name__))

    assert [tool.name for tool in selected] == [
        "coding_sandbox_execute_python",
        "database_sandbox_execute_sqlite",
    ]
