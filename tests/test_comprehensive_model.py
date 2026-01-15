"""
Comprehensive Model-Based Tests for Agent Evaluation

Extended model-based evaluation with additional dimensions and rubrics.
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Optional

from llama_index.llms.openai import OpenAI


@dataclass
class ComprehensiveEvalResult:
    """Result from comprehensive evaluation."""
    dimension: str
    score: float
    passed: bool
    feedback: str
    details: dict = None


class CompletenessGrader:
    """
    Evaluates if the response comprehensively addresses all aspects of the query.

    Rubric:
    - 5: Addresses all aspects thoroughly
    - 4: Addresses most aspects, minor gaps
    - 3: Addresses main points, some aspects missing
    - 2: Partial coverage, significant gaps
    - 1: Minimal coverage
    - 0: Does not address the query
    """

    def __init__(self, model: str = "gpt-4o", threshold: float = 3.0):
        self.llm = OpenAI(model=model, temperature=0)
        self.threshold = threshold

    async def grade(self, query: str, response: str) -> ComprehensiveEvalResult:
        prompt = f"""Evaluate how completely this response addresses the query.

QUERY: {query}

RESPONSE: {response}

RUBRIC:
- 5: Addresses all aspects thoroughly
- 4: Addresses most aspects, minor gaps
- 3: Addresses main points, some aspects missing
- 2: Partial coverage, significant gaps
- 1: Minimal coverage
- 0: Does not address the query

Provide:
SCORE: [0-5]
ASPECTS_COVERED: [list aspects that were addressed]
ASPECTS_MISSING: [list aspects that were not addressed]
FEEDBACK: [brief explanation]
"""

        try:
            result = await self.llm.acomplete(prompt)
            text = result.text

            import re
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text)
            score = float(score_match.group(1)) if score_match else 0

            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=\n[A-Z]|\Z)', text, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else text

            return ComprehensiveEvalResult(
                dimension="completeness",
                score=score,
                passed=score >= self.threshold,
                feedback=feedback,
                details={"raw_response": text},
            )
        except Exception as e:
            return ComprehensiveEvalResult(
                dimension="completeness",
                score=0,
                passed=False,
                feedback=f"Error: {str(e)}",
            )


class ClarityGrader:
    """
    Evaluates if the response is clear, well-organized, and easy to understand.

    Rubric:
    - 5: Exceptionally clear, well-structured, easy to follow
    - 4: Clear and organized, minor improvements possible
    - 3: Adequately clear, some organization issues
    - 2: Somewhat unclear or disorganized
    - 1: Difficult to understand
    - 0: Incomprehensible
    """

    def __init__(self, model: str = "gpt-4o", threshold: float = 3.0):
        self.llm = OpenAI(model=model, temperature=0)
        self.threshold = threshold

    async def grade(self, response: str) -> ComprehensiveEvalResult:
        prompt = f"""Evaluate the clarity and organization of this response.

RESPONSE: {response}

RUBRIC:
- 5: Exceptionally clear, well-structured, easy to follow
- 4: Clear and organized, minor improvements possible
- 3: Adequately clear, some organization issues
- 2: Somewhat unclear or disorganized
- 1: Difficult to understand
- 0: Incomprehensible

Provide:
SCORE: [0-5]
STRENGTHS: [what makes it clear]
WEAKNESSES: [what could be clearer]
FEEDBACK: [brief explanation]
"""

        try:
            result = await self.llm.acomplete(prompt)
            text = result.text

            import re
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text)
            score = float(score_match.group(1)) if score_match else 0

            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=\n[A-Z]|\Z)', text, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else text

            return ComprehensiveEvalResult(
                dimension="clarity",
                score=score,
                passed=score >= self.threshold,
                feedback=feedback,
            )
        except Exception as e:
            return ComprehensiveEvalResult(
                dimension="clarity",
                score=0,
                passed=False,
                feedback=f"Error: {str(e)}",
            )


class ActionabilityGrader:
    """
    Evaluates if the response provides actionable information for claims processing.

    Rubric:
    - 5: Highly actionable, clear next steps
    - 4: Actionable with minor clarifications needed
    - 3: Somewhat actionable, requires interpretation
    - 2: Limited actionability
    - 1: Not actionable
    - 0: Counterproductive or misleading
    """

    def __init__(self, model: str = "gpt-4o", threshold: float = 3.0):
        self.llm = OpenAI(model=model, temperature=0)
        self.threshold = threshold

    async def grade(self, query: str, response: str) -> ComprehensiveEvalResult:
        prompt = f"""Evaluate if this response provides actionable information for insurance claims processing.

QUERY: {query}

RESPONSE: {response}

RUBRIC:
- 5: Highly actionable, clear next steps
- 4: Actionable with minor clarifications needed
- 3: Somewhat actionable, requires interpretation
- 2: Limited actionability
- 1: Not actionable
- 0: Counterproductive or misleading

Provide:
SCORE: [0-5]
ACTIONABLE_ITEMS: [list specific actions that can be taken]
FEEDBACK: [brief explanation]
"""

        try:
            result = await self.llm.acomplete(prompt)
            text = result.text

            import re
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text)
            score = float(score_match.group(1)) if score_match else 0

            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=\n[A-Z]|\Z)', text, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else text

            return ComprehensiveEvalResult(
                dimension="actionability",
                score=score,
                passed=score >= self.threshold,
                feedback=feedback,
            )
        except Exception as e:
            return ComprehensiveEvalResult(
                dimension="actionability",
                score=0,
                passed=False,
                feedback=f"Error: {str(e)}",
            )


class ConsistencyGrader:
    """
    Evaluates if the response is internally consistent (no contradictions).

    Rubric:
    - 5: Perfectly consistent throughout
    - 4: Consistent with negligible issues
    - 3: Mostly consistent, minor discrepancies
    - 2: Some contradictions or inconsistencies
    - 1: Significant contradictions
    - 0: Completely contradictory
    """

    def __init__(self, model: str = "gpt-4o", threshold: float = 3.5):
        self.llm = OpenAI(model=model, temperature=0)
        self.threshold = threshold

    async def grade(self, response: str) -> ComprehensiveEvalResult:
        prompt = f"""Evaluate the internal consistency of this response. Check for contradictions.

RESPONSE: {response}

RUBRIC:
- 5: Perfectly consistent throughout
- 4: Consistent with negligible issues
- 3: Mostly consistent, minor discrepancies
- 2: Some contradictions or inconsistencies
- 1: Significant contradictions
- 0: Completely contradictory

Provide:
SCORE: [0-5]
CONTRADICTIONS: [list any contradictions found, or "None"]
FEEDBACK: [brief explanation]
"""

        try:
            result = await self.llm.acomplete(prompt)
            text = result.text

            import re
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text)
            score = float(score_match.group(1)) if score_match else 0

            feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=\n[A-Z]|\Z)', text, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else text

            return ComprehensiveEvalResult(
                dimension="consistency",
                score=score,
                passed=score >= self.threshold,
                feedback=feedback,
            )
        except Exception as e:
            return ComprehensiveEvalResult(
                dimension="consistency",
                score=0,
                passed=False,
                feedback=f"Error: {str(e)}",
            )


async def run_agent_and_get_response(agent, query: str) -> str:
    """Run agent and return response."""
    handler = agent.run(user_msg=query)
    response = await handler
    return str(response)


class TestCompletenessGrader:
    """Tests for completeness evaluation."""

    @pytest.mark.asyncio
    async def test_needle_completeness(self, initialized_agent, needle_cases):
        """Needle queries should be completely answered."""
        grader = CompletenessGrader(threshold=3.5)
        test_case = needle_cases[0]

        response = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(test_case.query, response)

        print("\n" + "="*60)
        print("COMPLETENESS EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"Response: {response[:300]}...")
        print(f"Score: {result.score}/5.0")
        print(f"Passed: {result.passed}")
        print(f"Feedback: {result.feedback}")

        assert result.passed, f"Completeness below threshold: {result.feedback}"

    @pytest.mark.asyncio
    async def test_summary_completeness(self, initialized_agent, summary_cases):
        """Summary queries should cover all key aspects."""
        grader = CompletenessGrader(threshold=3.0)
        test_case = summary_cases[0]

        response = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(test_case.query, response)

        print("\n" + "="*60)
        print("SUMMARY COMPLETENESS")
        print("="*60)
        print(f"Score: {result.score}/5.0")
        print(f"Feedback: {result.feedback}")

        assert result.passed, f"Summary not complete enough: {result.feedback}"


class TestClarityGrader:
    """Tests for clarity evaluation."""

    @pytest.mark.asyncio
    async def test_response_clarity(self, initialized_agent, test_cases):
        """All responses should be clear and well-organized."""
        grader = ClarityGrader(threshold=3.0)

        results = []
        for tc in test_cases[:3]:
            response = await run_agent_and_get_response(initialized_agent, tc.query)
            result = await grader.grade(response)
            results.append((tc.query[:40], result))

        print("\n" + "="*60)
        print("CLARITY EVALUATION")
        print("="*60)

        for query, result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n{query}...")
            print(f"  Score: {result.score}/5.0 [{status}]")
            print(f"  Feedback: {result.feedback[:100]}...")

        # Most should pass
        pass_rate = sum(1 for _, r in results if r.passed) / len(results)
        assert pass_rate >= 0.66, f"Clarity pass rate too low: {pass_rate*100:.0f}%"


class TestActionabilityGrader:
    """Tests for actionability evaluation."""

    @pytest.mark.asyncio
    async def test_mcp_actionability(self, initialized_agent, mcp_cases):
        """MCP responses should be actionable for claims processing."""
        grader = ActionabilityGrader(threshold=3.0)
        test_case = mcp_cases[0]

        response = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(test_case.query, response)

        print("\n" + "="*60)
        print("ACTIONABILITY EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"Response: {response[:300]}...")
        print(f"Score: {result.score}/5.0")
        print(f"Feedback: {result.feedback}")

        assert result.passed, f"Not actionable enough: {result.feedback}"


class TestConsistencyGrader:
    """Tests for consistency evaluation."""

    @pytest.mark.asyncio
    async def test_response_consistency(self, initialized_agent, summary_cases):
        """Responses should be internally consistent."""
        grader = ConsistencyGrader(threshold=3.5)
        test_case = summary_cases[0]

        response = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(response)

        print("\n" + "="*60)
        print("CONSISTENCY EVALUATION")
        print("="*60)
        print(f"Response: {response[:300]}...")
        print(f"Score: {result.score}/5.0")
        print(f"Feedback: {result.feedback}")

        assert result.passed, f"Inconsistencies found: {result.feedback}"


class TestComprehensiveEvaluation:
    """Combined comprehensive evaluation."""

    @pytest.mark.asyncio
    async def test_full_comprehensive_eval(self, initialized_agent, test_cases):
        """Run all comprehensive evaluations on a test case."""
        test_case = test_cases[0]

        response = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        # Run all graders
        completeness = await CompletenessGrader().grade(test_case.query, response)
        clarity = await ClarityGrader().grade(response)
        actionability = await ActionabilityGrader().grade(test_case.query, response)
        consistency = await ConsistencyGrader().grade(response)

        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*70)
        print(f"Query: {test_case.query}")
        print(f"\nResponse: {response[:200]}...")
        print("\n" + "-"*70)

        results = [
            ("Completeness", completeness),
            ("Clarity", clarity),
            ("Actionability", actionability),
            ("Consistency", consistency),
        ]

        all_passed = True
        total_score = 0
        for name, result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{name:<15}: {result.score:.1f}/5.0 [{status}]")
            total_score += result.score
            if not result.passed:
                all_passed = False

        avg_score = total_score / len(results)
        print("-"*70)
        print(f"Average Score: {avg_score:.2f}/5.0")
        print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")

        # Average should be above 3.0
        assert avg_score >= 3.0, f"Average score too low: {avg_score:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
