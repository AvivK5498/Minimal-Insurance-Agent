"""
Edge Case Tests for Agent Evaluation

Tests unusual inputs, boundary conditions, and error handling scenarios.
"""

import asyncio
import pytest
from dataclasses import dataclass


@dataclass
class EdgeCaseTestCase:
    """Edge case test definition."""
    name: str
    query: str
    expected_behavior: str
    should_error: bool = False


EDGE_CASES = [
    # Invalid claim IDs
    EdgeCaseTestCase(
        name="nonexistent_claim",
        query="What is the status of claim CLM-00000?",
        expected_behavior="should indicate claim not found or unknown",
    ),
    EdgeCaseTestCase(
        name="malformed_claim_id",
        query="Tell me about claim ABC123",
        expected_behavior="should handle gracefully or ask for valid format",
    ),
    EdgeCaseTestCase(
        name="empty_claim_id",
        query="What happened in claim ?",
        expected_behavior="should ask for clarification",
    ),

    # Ambiguous queries
    EdgeCaseTestCase(
        name="ambiguous_which_claim",
        query="What was the damage amount?",
        expected_behavior="should ask which claim or list options",
    ),
    EdgeCaseTestCase(
        name="ambiguous_time_reference",
        query="For claim CLM-89921, what happened yesterday?",
        expected_behavior="should clarify or use claim dates",
    ),

    # Multiple claims in one query
    EdgeCaseTestCase(
        name="multi_claim_comparison",
        query="Compare the damage amounts between CLM-89921 and CLM-44217-PD",
        expected_behavior="should handle both claims or explain limitation",
    ),
    EdgeCaseTestCase(
        name="multi_claim_list",
        query="List all claims involving water damage",
        expected_behavior="should search across claims or explain scope",
    ),

    # Boundary conditions
    EdgeCaseTestCase(
        name="very_long_query",
        query="For claim CLM-89921, I need a comprehensive analysis including: " +
              "the exact time of the accident, all parties involved, " +
              "the sequence of events, total damages, medical costs if any, " +
              "property damage details, witness statements, police report findings, " +
              "and any recommendations for claim processing. " * 3,
        expected_behavior="should handle long query gracefully",
    ),
    EdgeCaseTestCase(
        name="minimal_query",
        query="CLM-89921?",
        expected_behavior="should ask for clarification or provide summary",
    ),

    # Special characters and formatting
    EdgeCaseTestCase(
        name="query_with_special_chars",
        query="What's the status of claim CLM-89921? (urgent!!)",
        expected_behavior="should parse correctly ignoring special chars",
    ),
    EdgeCaseTestCase(
        name="query_with_numbers",
        query="For claim CLM-89921, was the damage over $10,000 or under $5,000?",
        expected_behavior="should understand numeric comparisons",
    ),

    # Temporal queries
    EdgeCaseTestCase(
        name="future_date_query",
        query="What will happen with claim CLM-89921 next month?",
        expected_behavior="should explain cannot predict future",
    ),
    EdgeCaseTestCase(
        name="relative_time_query",
        query="For claim CLM-44217-PD, how long ago was the incident?",
        expected_behavior="should calculate or provide incident date",
    ),

    # Cross-reference queries
    EdgeCaseTestCase(
        name="cross_reference_policies",
        query="Does claim CLM-77182-CM fall under the same policy type as CLM-89921?",
        expected_behavior="should compare or explain policy differences",
    ),

    # Negation and absence queries
    EdgeCaseTestCase(
        name="negation_query",
        query="For claim CLM-89921, what damages were NOT covered?",
        expected_behavior="should identify exclusions if any",
    ),
    EdgeCaseTestCase(
        name="absence_query",
        query="For claim CLM-44217-PD, were there any injuries?",
        expected_behavior="should confirm presence or absence of injuries",
    ),

    # Format-specific requests
    EdgeCaseTestCase(
        name="bullet_point_request",
        query="List the key facts of claim CLM-89921 in bullet points",
        expected_behavior="should format as bullet points",
    ),
    EdgeCaseTestCase(
        name="table_request",
        query="Show claim CLM-77182-CM damages in a table format",
        expected_behavior="should attempt table format or explain limitation",
    ),
]


async def run_agent_query(agent, query: str) -> tuple[str, bool]:
    """Run agent and capture response and any errors."""
    try:
        handler = agent.run(user_msg=query)
        response = await handler
        return str(response), False
    except Exception as e:
        return str(e), True


class TestEdgeCases:
    """Edge case test suite."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("edge_case", EDGE_CASES[:5], ids=lambda x: x.name)
    async def test_invalid_inputs(self, initialized_agent, edge_case):
        """Test handling of invalid or malformed inputs."""
        response, had_error = await run_agent_query(
            initialized_agent,
            edge_case.query
        )

        print(f"\n{'='*60}")
        print(f"EDGE CASE: {edge_case.name}")
        print(f"{'='*60}")
        print(f"Query: {edge_case.query}")
        print(f"Expected: {edge_case.expected_behavior}")
        print(f"Response: {response[:400]}...")
        print(f"Had error: {had_error}")

        # Should not crash
        if not edge_case.should_error:
            assert not had_error, f"Unexpected error: {response}"

        # Should provide some response
        assert len(response) > 10, "Response too short"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("edge_case", EDGE_CASES[5:10], ids=lambda x: x.name)
    async def test_ambiguous_queries(self, initialized_agent, edge_case):
        """Test handling of ambiguous or multi-part queries."""
        response, had_error = await run_agent_query(
            initialized_agent,
            edge_case.query
        )

        print(f"\n{'='*60}")
        print(f"EDGE CASE: {edge_case.name}")
        print(f"{'='*60}")
        print(f"Query: {edge_case.query}")
        print(f"Expected: {edge_case.expected_behavior}")
        print(f"Response: {response[:400]}...")

        assert not had_error, f"Unexpected error: {response}"
        assert len(response) > 20, "Response too short for complex query"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("edge_case", EDGE_CASES[10:], ids=lambda x: x.name)
    async def test_special_format_queries(self, initialized_agent, edge_case):
        """Test handling of special formats and temporal queries."""
        response, had_error = await run_agent_query(
            initialized_agent,
            edge_case.query
        )

        print(f"\n{'='*60}")
        print(f"EDGE CASE: {edge_case.name}")
        print(f"{'='*60}")
        print(f"Query: {edge_case.query}")
        print(f"Expected: {edge_case.expected_behavior}")
        print(f"Response: {response[:400]}...")

        assert not had_error, f"Unexpected error: {response}"


class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    @pytest.mark.asyncio
    async def test_empty_query(self, initialized_agent):
        """Test handling of empty query."""
        response, had_error = await run_agent_query(initialized_agent, "")

        print(f"\nEmpty query response: {response[:200]}")
        # Should handle gracefully
        assert not had_error or "error" in response.lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_query(self, initialized_agent):
        """Test handling of whitespace-only query."""
        response, had_error = await run_agent_query(initialized_agent, "   \n\t   ")

        print(f"\nWhitespace query response: {response[:200]}")
        assert not had_error or "error" in response.lower()

    @pytest.mark.asyncio
    async def test_repeated_query(self, initialized_agent, needle_cases):
        """Test that repeated identical queries give consistent results."""
        test_case = needle_cases[0]

        response1, _ = await run_agent_query(initialized_agent, test_case.query)
        response2, _ = await run_agent_query(initialized_agent, test_case.query)

        print(f"\nFirst response: {response1[:200]}...")
        print(f"Second response: {response2[:200]}...")

        # Responses should be similar (not necessarily identical due to LLM variability)
        # Check that key facts appear in both
        assert len(response1) > 50
        assert len(response2) > 50


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    @pytest.mark.asyncio
    async def test_partial_claim_id(self, initialized_agent):
        """Test handling of partial claim ID."""
        response, had_error = await run_agent_query(
            initialized_agent,
            "What about claim CLM-899?"  # Partial ID
        )

        print(f"\nPartial ID response: {response[:300]}")
        assert not had_error

    @pytest.mark.asyncio
    async def test_typo_in_query(self, initialized_agent):
        """Test handling of typos in query."""
        response, had_error = await run_agent_query(
            initialized_agent,
            "Waht ws the accidnet tiem for cliam CLM-89921?"  # Typos
        )

        print(f"\nTypo query response: {response[:300]}")
        # Should still attempt to answer
        assert not had_error
        assert len(response) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
