"""Unit tests for `app.agents.chat_modes.policy`."""

import pytest

from app.agents.chat_modes.policy import (
    AGENT_POLICY,
    INTERNAL_SEARCH_POLICY,
    WEB_SEARCH_POLICY,
    resolve_chat_mode_policy,
)


class TestResolveChatModePolicy:
    @pytest.mark.parametrize(
        ("wire_value", "expected"),
        [
            ("internal_search", INTERNAL_SEARCH_POLICY),
            ("web_search", WEB_SEARCH_POLICY),
            ("agent", AGENT_POLICY),
        ],
    )
    def test_canonical_values_resolve_directly(self, wire_value, expected) -> None:
        assert resolve_chat_mode_policy(wire_value) is expected

    @pytest.mark.parametrize(
        "wire_value",
        ["Agent", "AGENT", "  agent  ", "Web_Search", " web_search"],
    )
    def test_canonical_values_are_case_and_whitespace_insensitive(self, wire_value) -> None:
        normalized = wire_value.strip().lower()
        expected = {
            "agent": AGENT_POLICY,
            "web_search": WEB_SEARCH_POLICY,
        }[normalized]
        assert resolve_chat_mode_policy(wire_value) is expected

    @pytest.mark.parametrize(
        "wire_value",
        ["analysis", "deep_research", "creative", "precise", "standard", "quick"],
    )
    def test_legacy_values_fall_back_to_internal_search(self, wire_value) -> None:
        assert resolve_chat_mode_policy(wire_value) is INTERNAL_SEARCH_POLICY

    @pytest.mark.parametrize("wire_value", [None, "", "   ", "unknown_mode", "agent:quick"])
    def test_missing_or_unrecognized_values_fall_back_to_internal_search(self, wire_value) -> None:
        assert resolve_chat_mode_policy(wire_value) is INTERNAL_SEARCH_POLICY


class TestChatModePolicyShape:
    def test_internal_search_forces_knowledge_no_web_prefetches(self) -> None:
        assert INTERNAL_SEARCH_POLICY.has_knowledge is True
        assert INTERNAL_SEARCH_POLICY.include_web_search is False
        assert INTERNAL_SEARCH_POLICY.prefetch_retrieval is True

    def test_web_search_disables_knowledge_no_prefetch(self) -> None:
        assert WEB_SEARCH_POLICY.has_knowledge is False
        assert WEB_SEARCH_POLICY.include_web_search is True
        assert WEB_SEARCH_POLICY.prefetch_retrieval is False

    def test_agent_enables_both_tools_no_upfront_prefetch(self) -> None:
        assert AGENT_POLICY.has_knowledge is True
        assert AGENT_POLICY.include_web_search is True
        assert AGENT_POLICY.prefetch_retrieval is False

    def test_all_policies_use_the_quick_loop(self) -> None:
        for policy in (INTERNAL_SEARCH_POLICY, WEB_SEARCH_POLICY, AGENT_POLICY):
            assert policy.loop_chat_mode == "quick"

    def test_policy_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            INTERNAL_SEARCH_POLICY.has_knowledge = False  # type: ignore[misc]
