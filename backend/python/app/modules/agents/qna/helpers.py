"""Survivors of `qna/nodes.py` (deleted with the rest of LangGraph in the
agent-loop migration) still imported by the agent loop:

- `clean_tool_result` — `app.agents.agent_loop.tool_adapter`
- `_extract_web_records_from_tool_results` — `app.agents.agent_loop.respond`,
  `app.agents.agent_loop.answer_streamer`
- `_tool_names_and_results_from_state` — `app.agents.agent_loop.respond`

Kept as plain functions with no LangGraph/`ChatState` coupling beyond the
one type hint below (a `TypedDict`, not a LangGraph runtime type).
"""

from __future__ import annotations

import json
from typing import Any

from app.modules.agents.qna.chat_state import ChatState

TOOL_RESULT_TUPLE_LENGTH = 2

REMOVE_FIELDS = {
    "self", "_links", "_embedded", "_meta", "_metadata",
    "expand", "expansions", "schema", "$schema",
    "avatarUrls", "avatarUrl", "iconUrl", "iconUri", "thumbnailUrl",
    "avatar", "icon", "thumbnail", "profilePicture",
    "trace", "traceId", "requestId", "correlationId",
    "debug", "debugInfo", "stack", "stackTrace",
    "headers", "cookies", "request", "response",
    "httpVersion", "protocol", "encoding",
    "timeZone", "timezone", "locale", "language",
    "accountType", "active", "properties",
    "hierarchyLevel", "subtask", "avatarId",
    "watches", "votes", "watchers", "voters",
    "changelog", "history", "worklog", "worklogs",
}


def clean_tool_result(result: object) -> object:
    """Clean tool result by removing verbose fields."""
    if isinstance(result, tuple) and len(result) == TOOL_RESULT_TUPLE_LENGTH:
        success, data = result
        return (success, clean_tool_result(data))

    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            cleaned = clean_tool_result(parsed)
            return json.dumps(cleaned, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return result

    if isinstance(result, dict):
        cleaned = {}
        for key, value in result.items():
            if key in REMOVE_FIELDS or key.lower() in REMOVE_FIELDS:
                continue
            if key.startswith(("_", "$")):
                continue

            if isinstance(value, dict):
                cleaned_value = clean_tool_result(value)
                if cleaned_value:
                    cleaned[key] = cleaned_value
            elif isinstance(value, list):
                cleaned[key] = [clean_tool_result(item) for item in value]
            else:
                cleaned[key] = value
        return cleaned

    if isinstance(result, list):
        return [clean_tool_result(item) for item in result]

    return result


def _extract_web_records_from_tool_results(
    tool_results: list[dict], org_id: str,
) -> list[dict]:
    """Build web_records from web_search / fetch_url tool results that were
    executed in the agent's execution phase (before the respond step).

    Delegates to `ToolHandlerRegistry.extract_records` — the same path that
    `streaming.py`'s `execute_tool_calls` uses for live tool output — so
    citation URLs are generated identically (text-fragment URLs for
    fetch_url blocks, plain links for web_search snippets).
    """
    from app.utils.tool_handlers import ToolHandlerRegistry

    web_records: list[dict] = []
    for r in tool_results:
        if r.get("status") != "success":
            continue
        result = r.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                continue
        if not isinstance(result, dict):
            continue
        handler = ToolHandlerRegistry.get_handler(result)
        for rec in handler.extract_records(result, org_id=org_id):
            if rec.get("source_type") == "web":
                web_records.append(rec)
    return web_records


def _tool_names_and_results_from_state(state: ChatState) -> dict[str, Any]:
    """Derive succeeded/failed tool names and full tool results from state (no separate state fields)."""
    results = state.get("all_tool_results") or state.get("tool_results") or []
    succeeded = [r.get("tool_name") for r in results if r.get("tool_name") and r.get("status") == "success"]
    failed = [r.get("tool_name") for r in results if r.get("tool_name") and r.get("status") == "error"]
    return {
        "succeeded_tool_names": succeeded,
        "failed_tool_names": failed,
        "tool_results": results,
    }


__all__ = [
    "REMOVE_FIELDS",
    "TOOL_RESULT_TUPLE_LENGTH",
    "_extract_web_records_from_tool_results",
    "_tool_names_and_results_from_state",
    "clean_tool_result",
]
