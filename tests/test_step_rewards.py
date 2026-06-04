"""Unit tests for PRM step rewards and budget-constrained penalties (§3.4).

Tests cover:
  - compute_step_rewards: valid tool_call blocks, invalid JSON, missing fields,
    known vs unknown tool names, discount factor, multiple steps
  - compute_budget_penalty: step count budget, token budget, infinite loop
    detection, edge cases
  - _blocks_near_identical: exact match, near match, mismatch
"""

import pytest
from grpo.rewards import (
    compute_step_rewards,
    compute_budget_penalty,
    _extract_tool_call_blocks,
    _blocks_near_identical,
)


# ==============================================================================
# Helpers to build test completions
# ==============================================================================

def _make_fenced_tool_call(name: str, arguments: dict) -> str:
    """Build a fenced tool_call block from name and arguments dict."""
    import json
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"```tool_call\n{payload}\n```"


def _make_xml_tool_call(name: str, arguments: dict) -> str:
    """Build an XML-style tool_call block."""
    import json
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>{payload}</tool_call>"


# ==============================================================================
# Tests for compute_step_rewards
# ==============================================================================

class TestComputeStepRewards:
    """Tests for the process-supervised step reward scorer."""

    def test_no_tool_calls_returns_zero(self):
        """Completion without tool_call blocks should return 0.0."""
        assert compute_step_rewards("Hello, I'm a regular response.") == 0.0

    def test_single_valid_known_tool(self):
        """A single valid call to a known tool: +0.3 (format) + 0.2 (known) = 0.5."""
        text = _make_fenced_tool_call("get_weather", {"city": "Paris"})
        reward = compute_step_rewards(text)
        assert abs(reward - 0.5) < 1e-6, f"Expected 0.5, got {reward}"

    def test_single_valid_unknown_but_plausible_tool(self):
        """A tool with a valid snake_case name not in the known set gets partial credit."""
        text = _make_fenced_tool_call("custom_tool_xyz", {"key": "value"})
        reward = compute_step_rewards(text)
        # +0.3 (format) + 0.1 (plausible identifier) = 0.4
        assert abs(reward - 0.4) < 1e-6, f"Expected 0.4, got {reward}"

    def test_single_valid_nonsense_name(self):
        """A tool name that isn't a valid snake_case identifier gets only format reward."""
        text = _make_fenced_tool_call("123-INVALID!", {"key": "value"})
        reward = compute_step_rewards(text)
        # +0.3 only (format valid, but name is nonsensical)
        assert abs(reward - 0.3) < 1e-6, f"Expected 0.3, got {reward}"

    def test_invalid_json(self):
        """Malformed JSON should return -0.5."""
        text = "```tool_call\n{this is not valid json}\n```"
        reward = compute_step_rewards(text)
        assert abs(reward - (-0.5)) < 1e-6, f"Expected -0.5, got {reward}"

    def test_missing_name_field(self):
        """JSON without 'name' or 'tool_name' field gets -0.5."""
        import json
        payload = json.dumps({"arguments": {"city": "London"}})
        text = f"```tool_call\n{payload}\n```"
        reward = compute_step_rewards(text)
        assert abs(reward - (-0.5)) < 1e-6, f"Expected -0.5, got {reward}"

    def test_missing_arguments_field(self):
        """JSON without 'arguments'/'params'/'parameters' field gets -0.5."""
        import json
        payload = json.dumps({"name": "get_weather"})
        text = f"```tool_call\n{payload}\n```"
        reward = compute_step_rewards(text)
        assert abs(reward - (-0.5)) < 1e-6, f"Expected -0.5, got {reward}"

    def test_xml_format_works(self):
        """XML-style <tool_call> tags should be parsed correctly."""
        text = _make_xml_tool_call("calculator", {"expression": "2+2"})
        reward = compute_step_rewards(text)
        assert abs(reward - 0.5) < 1e-6, f"Expected 0.5, got {reward}"

    def test_discount_factor_applied(self):
        """Multiple steps should apply gamma^t discounting."""
        call1 = _make_fenced_tool_call("get_weather", {"city": "Tokyo"})
        call2 = _make_fenced_tool_call("calculator", {"expression": "1+1"})
        text = f"First call:\n{call1}\nSecond call:\n{call2}"

        gamma = 0.95
        # Step 0: gamma^0 * 0.5 = 0.5
        # Step 1: gamma^1 * 0.5 = 0.475
        expected = 0.5 + gamma * 0.5
        reward = compute_step_rewards(text, gamma=gamma)
        assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"

    def test_mixed_valid_and_invalid_steps(self):
        """Mix of valid and invalid steps should sum correctly with discounting."""
        valid_call = _make_fenced_tool_call("web_search", {"query": "test"})
        invalid_call = "```tool_call\n{broken json\n```"
        text = f"{valid_call}\n{invalid_call}"

        gamma = 0.95
        # Step 0: gamma^0 * 0.5 = 0.5
        # Step 1: gamma^1 * (-0.5) = -0.475
        expected = 0.5 + gamma * (-0.5)
        reward = compute_step_rewards(text, gamma=gamma)
        assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"

    def test_custom_gamma(self):
        """Custom gamma=1.0 means no discounting."""
        call1 = _make_fenced_tool_call("translate", {"text": "hi", "target_language": "ja"})
        call2 = _make_fenced_tool_call("send_email", {"to": "a@b.com", "subject": "Hi", "body": "Hello"})
        text = f"{call1}\n{call2}"

        # Both known tools, both valid: 0.5 * 2 = 1.0
        reward = compute_step_rewards(text, gamma=1.0)
        assert abs(reward - 1.0) < 1e-6, f"Expected 1.0, got {reward}"

    def test_tool_name_field_alias(self):
        """'tool_name' alias should be accepted as well as 'name'."""
        import json
        payload = json.dumps({"tool_name": "get_weather", "params": {"city": "NYC"}})
        text = f"```tool_call\n{payload}\n```"
        reward = compute_step_rewards(text)
        # has_name via 'tool_name', has_args via 'params' → +0.3
        # tool_name "get_weather" is known → +0.2
        assert abs(reward - 0.5) < 1e-6, f"Expected 0.5, got {reward}"


# ==============================================================================
# Tests for compute_budget_penalty
# ==============================================================================

class TestComputeBudgetPenalty:
    """Tests for the budget-constrained penalty function."""

    def test_no_tool_calls_no_penalty(self):
        """Short completion with no tool calls should have zero penalty."""
        assert compute_budget_penalty("Hello world.") == 0.0

    def test_within_budget_no_penalty(self):
        """A few tool calls within budget should return 0.0."""
        calls = "\n".join(
            _make_fenced_tool_call("calculator", {"expression": f"{i}+1"})
            for i in range(3)
        )
        assert compute_budget_penalty(calls, max_steps=5) == 0.0

    def test_exceeds_step_budget(self):
        """More tool calls than max_steps should return -3.0."""
        calls = "\n".join(
            _make_fenced_tool_call("calculator", {"expression": f"{i}+{i}"})
            for i in range(7)  # 7 unique calls
        )
        penalty = compute_budget_penalty(calls, max_steps=5)
        assert penalty == -3.0, f"Expected -3.0, got {penalty}"

    def test_exceeds_token_budget(self):
        """Very long completion exceeding max_tokens should return -1.5."""
        # Build a long text (>6000 chars = >1500 estimated tokens)
        long_text = "A" * 6100
        penalty = compute_budget_penalty(long_text, max_tokens=1500)
        assert penalty == -1.5, f"Expected -1.5, got {penalty}"

    def test_infinite_loop_detection(self):
        """3+ consecutive identical tool calls should trigger -5.0 penalty."""
        same_call = _make_fenced_tool_call("get_weather", {"city": "London"})
        text = f"{same_call}\n{same_call}\n{same_call}"
        penalty = compute_budget_penalty(text, max_steps=10)
        assert penalty == -5.0, f"Expected -5.0, got {penalty}"

    def test_infinite_loop_near_identical(self):
        """3 consecutive near-identical calls (minor whitespace diff) trigger -5.0."""
        import json
        base = {"name": "calculator", "arguments": {"expression": "2+2"}}
        b1 = f"```tool_call\n{json.dumps(base)}\n```"
        b2 = f"```tool_call\n{json.dumps(base)}\n```"  # identical
        b3 = f"```tool_call\n {json.dumps(base)} \n```"  # minor whitespace
        text = f"{b1}\n{b2}\n{b3}"
        penalty = compute_budget_penalty(text, max_steps=10)
        assert penalty == -5.0, f"Expected -5.0, got {penalty}"

    def test_loop_priority_over_step_budget(self):
        """Infinite loop penalty (-5.0) takes priority over step budget (-3.0)."""
        same_call = _make_fenced_tool_call("web_search", {"query": "test"})
        # 6 identical calls: both loop AND step-budget violated
        text = "\n".join([same_call] * 6)
        penalty = compute_budget_penalty(text, max_steps=5)
        assert penalty == -5.0, f"Loop penalty should take priority, got {penalty}"

    def test_boundary_step_count(self):
        """Exactly max_steps calls should NOT trigger penalty."""
        calls = "\n".join(
            _make_fenced_tool_call("calculator", {"expression": f"{i}*2"})
            for i in range(5)  # exactly 5
        )
        penalty = compute_budget_penalty(calls, max_steps=5)
        assert penalty == 0.0, f"Boundary case should be 0.0, got {penalty}"

    def test_empty_string(self):
        """Empty completion should return 0.0."""
        assert compute_budget_penalty("") == 0.0


# ==============================================================================
# Tests for _extract_tool_call_blocks
# ==============================================================================

class TestExtractToolCallBlocks:
    """Tests for the tool_call block extraction utility."""

    def test_fenced_extraction(self):
        """Should extract JSON from fenced tool_call blocks."""
        text = '```tool_call\n{"name": "test", "arguments": {}}\n```'
        blocks = _extract_tool_call_blocks(text)
        assert len(blocks) == 1
        assert '"name": "test"' in blocks[0]

    def test_xml_extraction(self):
        """Should extract JSON from XML-style tool_call tags."""
        text = '<tool_call>{"name": "test", "arguments": {}}</tool_call>'
        blocks = _extract_tool_call_blocks(text)
        assert len(blocks) == 1

    def test_mixed_formats(self):
        """Should extract from both fenced and XML blocks in same text."""
        text = (
            '```tool_call\n{"name": "a", "arguments": {}}\n```\n'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        blocks = _extract_tool_call_blocks(text)
        assert len(blocks) == 2

    def test_no_blocks(self):
        """Text without tool_call blocks should return empty list."""
        assert _extract_tool_call_blocks("Just regular text.") == []


# ==============================================================================
# Tests for _blocks_near_identical
# ==============================================================================

class TestBlocksNearIdentical:
    """Tests for the near-identical block comparison helper."""

    def test_exact_match(self):
        assert _blocks_near_identical('{"a": 1}', '{"a": 1}') is True

    def test_whitespace_difference(self):
        assert _blocks_near_identical('{"a": 1}', '{ "a" : 1 }') is True

    def test_completely_different(self):
        assert _blocks_near_identical('{"a": 1}', '{"b": 99, "c": "xyz"}') is False

    def test_empty_strings(self):
        assert _blocks_near_identical("", "") is True  # both empty after strip
        # Actually both become empty, exact match

    def test_one_empty(self):
        assert _blocks_near_identical("", '{"a": 1}') is False
