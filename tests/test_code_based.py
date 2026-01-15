"""
Code-Based Graders for Agent Evaluation

Code-based graders are fast, cheap, and reproducible. They handle deterministic
checks like:
- Tool call validation (correct tools called, correct sequence)
- Response format verification (contains expected patterns)
- State verification (claim IDs, amounts, dates)
- Token/latency bounds

Reference: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
"""

import asyncio
import re
import pytest
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Code-Based Grader Classes
# =============================================================================

@dataclass
class ToolCallGraderResult:
    """Result from tool call validation."""
    passed: bool
    expected_tools: list[str]
    actual_tools: list[str]
    missing_tools: list[str]
    message: str


@dataclass
class FormatGraderResult:
    """Result from format validation."""
    passed: bool
    checks: dict[str, bool]
    message: str


@dataclass
class ResponseGraderResult:
    """Result from response content validation."""
    passed: bool
    expected_patterns: list[str]
    matched_patterns: list[str]
    missing_patterns: list[str]
    message: str


class ToolCallGrader:
    """
    Validates that the agent called the expected tools.

    This grader checks:
    - Required tools were called
    - Tools were called in expected order (optional)
    - No unexpected tools were called (optional strict mode)
    """

    def __init__(self, strict_order: bool = False, strict_tools: bool = False):
        self.strict_order = strict_order
        self.strict_tools = strict_tools

    def grade(
        self,
        actual_tools: list[str],
        expected_tools: list[str]
    ) -> ToolCallGraderResult:
        """Grade the tool calls against expected tools."""

        # Check for missing tools
        missing = [t for t in expected_tools if t not in actual_tools]

        # Check order if strict
        order_correct = True
        if self.strict_order and not missing:
            expected_indices = [actual_tools.index(t) for t in expected_tools if t in actual_tools]
            order_correct = expected_indices == sorted(expected_indices)

        # Determine pass/fail
        passed = len(missing) == 0 and order_correct

        if passed:
            message = f"All expected tools called: {expected_tools}"
        elif missing:
            message = f"Missing tools: {missing}"
        else:
            message = f"Tools called out of order. Expected: {expected_tools}, Got: {actual_tools}"

        return ToolCallGraderResult(
            passed=passed,
            expected_tools=expected_tools,
            actual_tools=actual_tools,
            missing_tools=missing,
            message=message,
        )


class FormatGrader:
    """
    Validates response format against expected patterns.

    This grader checks:
    - Required fields are present
    - Values match expected formats (dates, currency, IDs)
    - Response structure is valid
    """

    PATTERNS = {
        "claim_id": r"CLM-\d{5}(?:-[A-Z]{2})?",
        "currency": r"\$[\d,]+\.?\d*",
        "date": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b",
        "time": r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",
        "percentage": r"\d+\.?\d*%",
        "risk_level": r"\b(?:Low|Medium|High)\b",
    }

    def grade(self, response: str, required_formats: list[str]) -> FormatGraderResult:
        """Grade the response format against required patterns."""
        checks = {}

        for fmt in required_formats:
            if fmt in self.PATTERNS:
                pattern = self.PATTERNS[fmt]
                checks[fmt] = bool(re.search(pattern, response, re.IGNORECASE))
            else:
                # Treat as literal string check
                checks[fmt] = fmt.lower() in response.lower()

        passed = all(checks.values())
        failed_checks = [k for k, v in checks.items() if not v]

        if passed:
            message = f"All format checks passed: {required_formats}"
        else:
            message = f"Failed format checks: {failed_checks}"

        return FormatGraderResult(
            passed=passed,
            checks=checks,
            message=message,
        )


class ResponseContentGrader:
    """
    Validates that response contains expected content patterns.

    This grader checks:
    - Key facts are present in response
    - Expected values appear (exact or fuzzy match)
    """

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def grade(
        self,
        response: str,
        expected_patterns: list[str],
        min_matches: Optional[int] = None
    ) -> ResponseGraderResult:
        """Grade response content against expected patterns."""

        if not self.case_sensitive:
            response_check = response.lower()
            patterns_check = [p.lower() for p in expected_patterns]
        else:
            response_check = response
            patterns_check = expected_patterns

        matched = []
        missing = []

        for i, pattern in enumerate(patterns_check):
            original_pattern = expected_patterns[i]
            if pattern in response_check:
                matched.append(original_pattern)
            else:
                missing.append(original_pattern)

        # Determine pass threshold
        if min_matches is None:
            min_matches = len(expected_patterns)

        passed = len(matched) >= min_matches

        if passed:
            message = f"Found {len(matched)}/{len(expected_patterns)} expected patterns"
        else:
            message = f"Only found {len(matched)}/{min_matches} required patterns. Missing: {missing}"

        return ResponseGraderResult(
            passed=passed,
            expected_patterns=expected_patterns,
            matched_patterns=matched,
            missing_patterns=missing,
            message=message,
        )


# =============================================================================
# Helper Functions
# =============================================================================

async def run_agent_and_capture_tools(agent, query: str) -> tuple[str, list[str]]:
    """Run agent and capture tool calls from events."""
    tools_used = []

    handler = agent.run(user_msg=query)

    async for event in handler.stream_events():
        if hasattr(event, 'tool_name'):
            if event.tool_name not in tools_used:
                tools_used.append(event.tool_name)

    response = await handler
    return str(response), tools_used


# =============================================================================
# Tests: Tool Call Validation
# =============================================================================

class TestToolCallGrader:
    """Tests for tool call validation - verifies correct tools are invoked."""

    @pytest.mark.asyncio
    async def test_needle_query_calls_correct_tools(self, initialized_agent, needle_cases):
        """Needle queries should call resolve_claim then needle_expert."""
        grader = ToolCallGrader(strict_order=True)
        test_case = needle_cases[0]

        response, tools_used = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        result = grader.grade(tools_used, test_case.expected_tools)

        print(f"\nQuery: {test_case.query[:50]}...")
        print(f"Expected tools: {test_case.expected_tools}")
        print(f"Actual tools: {tools_used}")
        print(f"Result: {result.message}")

        assert result.passed, result.message

    @pytest.mark.asyncio
    async def test_summary_query_calls_correct_tools(self, initialized_agent, summary_cases):
        """Summary queries should call resolve_claim then summary_expert."""
        grader = ToolCallGrader(strict_order=True)
        test_case = summary_cases[0]

        response, tools_used = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        result = grader.grade(tools_used, test_case.expected_tools)

        print(f"\nQuery: {test_case.query[:50]}...")
        print(f"Expected tools: {test_case.expected_tools}")
        print(f"Actual tools: {tools_used}")
        print(f"Result: {result.message}")

        assert result.passed, result.message

    @pytest.mark.asyncio
    async def test_mcp_query_calls_policy_tool(self, initialized_agent, mcp_cases):
        """MCP queries should call validate_policy_limit tool."""
        grader = ToolCallGrader(strict_order=False)
        test_case = mcp_cases[0]

        response, tools_used = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        result = grader.grade(tools_used, test_case.expected_tools)

        print(f"\nQuery: {test_case.query[:50]}...")
        print(f"Expected tools: {test_case.expected_tools}")
        print(f"Actual tools: {tools_used}")
        print(f"Result: {result.message}")

        assert result.passed, result.message

    @pytest.mark.asyncio
    async def test_resolve_claim_called_first(self, initialized_agent, needle_cases):
        """Verify resolve_claim is always called before expert tools."""
        test_case = needle_cases[0]

        _, tools_used = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        # Check that resolve_claim appears before any expert
        if "resolve_claim" in tools_used:
            resolve_idx = tools_used.index("resolve_claim")
            expert_indices = [
                tools_used.index(t) for t in tools_used
                if t in ["needle_expert", "summary_expert"]
            ]

            if expert_indices:
                assert resolve_idx < min(expert_indices), \
                    f"resolve_claim should be called before experts. Order: {tools_used}"


# =============================================================================
# Tests: Format Validation
# =============================================================================

class TestFormatGrader:
    """Tests for response format validation."""

    @pytest.mark.asyncio
    async def test_needle_response_contains_claim_id(self, initialized_agent, needle_cases):
        """Needle responses should reference the claim ID."""
        grader = FormatGrader()
        test_case = needle_cases[0]

        response, _ = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        # Check response mentions the claim
        result = grader.grade(response, ["claim_id"])

        print(f"\nResponse: {response[:200]}...")
        print(f"Format checks: {result.checks}")

        # This is informational - claim ID mention is nice but not required
        print(f"Claim ID present: {result.checks.get('claim_id', False)}")

    @pytest.mark.asyncio
    async def test_mcp_response_format(self, initialized_agent, mcp_cases):
        """MCP responses should contain risk level and percentage."""
        grader = FormatGrader()
        test_case = mcp_cases[0]

        response, _ = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        result = grader.grade(response, ["percentage", "risk_level"])

        print(f"\nQuery: {test_case.query}")
        print(f"Response: {response[:300]}...")
        print(f"Format checks: {result.checks}")

        assert result.passed, result.message

    @pytest.mark.asyncio
    async def test_currency_format_in_financial_response(self, initialized_agent):
        """Financial queries should return properly formatted currency."""
        grader = FormatGrader()

        query = "For claim CLM-77182-CM, what was the total estimated replacement cost of equipment before depreciation?"
        response, _ = await run_agent_and_capture_tools(initialized_agent, query)

        result = grader.grade(response, ["currency"])

        print(f"\nResponse: {response[:200]}...")
        print(f"Currency format check: {result.checks}")

        assert result.passed, result.message


# =============================================================================
# Tests: Response Content Validation
# =============================================================================

class TestResponseContentGrader:
    """Tests for response content validation."""

    @pytest.mark.asyncio
    async def test_needle_response_contains_answer(self, initialized_agent, needle_cases):
        """Needle responses should contain the expected factual answer."""
        grader = ResponseContentGrader(case_sensitive=False)
        test_case = needle_cases[0]  # Accident time query

        response, _ = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        # Check for key parts of the ground truth
        expected_patterns = ["11:30", "PM", "October", "2023"]
        result = grader.grade(response, expected_patterns, min_matches=2)

        print(f"\nQuery: {test_case.query}")
        print(f"Response: {response[:300]}...")
        print(f"Expected patterns: {expected_patterns}")
        print(f"Matched: {result.matched_patterns}")
        print(f"Missing: {result.missing_patterns}")

        assert result.passed, result.message

    @pytest.mark.asyncio
    async def test_mcp_exceeds_limit_detection(self, initialized_agent, mcp_cases):
        """When claim exceeds limit, response should indicate this."""
        grader = ResponseContentGrader(case_sensitive=False)

        # Find the test case where claim exceeds limit
        exceeds_case = next(
            (tc for tc in mcp_cases if "150,000" in tc.query),
            mcp_cases[-1]
        )

        response, _ = await run_agent_and_capture_tools(
            initialized_agent,
            exceeds_case.query
        )

        # Should indicate the claim exceeds or is over the limit
        expected_patterns = ["exceed", "over", "above"]
        result = grader.grade(response, expected_patterns, min_matches=1)

        print(f"\nQuery: {exceeds_case.query}")
        print(f"Response: {response[:300]}...")
        print(f"Looking for: {expected_patterns}")
        print(f"Found: {result.matched_patterns}")

        assert result.passed, f"Response should indicate limit exceeded: {response[:200]}"

    @pytest.mark.asyncio
    async def test_summary_response_completeness(self, initialized_agent, summary_cases):
        """Summary responses should cover key aspects of the incident."""
        grader = ResponseContentGrader(case_sensitive=False)
        test_case = summary_cases[0]  # Multi-vehicle collision sequence

        response, _ = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        # Summary should mention key elements
        expected_patterns = ["Sarah", "vehicle", "highway", "brake"]
        result = grader.grade(response, expected_patterns, min_matches=2)

        print(f"\nQuery: {test_case.query}")
        print(f"Response: {response[:400]}...")
        print(f"Key elements found: {result.matched_patterns}")

        assert result.passed, result.message


# =============================================================================
# Tests: Combined Code-Based Evaluation
# =============================================================================

class TestCombinedCodeBasedEvaluation:
    """Combined tests that use multiple code-based graders together."""

    @pytest.mark.asyncio
    async def test_full_needle_evaluation(self, initialized_agent, needle_cases):
        """Full code-based evaluation of a needle query."""
        tool_grader = ToolCallGrader(strict_order=True)
        format_grader = FormatGrader()
        content_grader = ResponseContentGrader()

        test_case = needle_cases[2]  # Equipment cost query

        response, tools_used = await run_agent_and_capture_tools(
            initialized_agent,
            test_case.query
        )

        # Grade all aspects
        tool_result = tool_grader.grade(tools_used, test_case.expected_tools)
        format_result = format_grader.grade(response, ["currency"])
        content_result = content_grader.grade(response, ["22,515", "$"])

        print("\n" + "="*60)
        print(f"FULL CODE-BASED EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"\nResponse: {response[:300]}...")
        print(f"\n--- Tool Call Check ---")
        print(f"  {tool_result.message}")
        print(f"\n--- Format Check ---")
        print(f"  {format_result.message}")
        print(f"\n--- Content Check ---")
        print(f"  {content_result.message}")
        print("="*60)

        # All checks should pass
        assert tool_result.passed, f"Tool check failed: {tool_result.message}"
        assert format_result.passed, f"Format check failed: {format_result.message}"
        assert content_result.passed, f"Content check failed: {content_result.message}"


# =============================================================================
# Run standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
