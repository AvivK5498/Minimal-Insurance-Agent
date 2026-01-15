"""
Model-Based Graders for Agent Evaluation

Model-based graders use LLMs to evaluate agent responses. They handle nuance
and open-ended tasks that code-based graders cannot assess. Key principles:
- Use clear rubrics for consistent evaluation
- Isolate judges for each dimension (correctness, relevancy, faithfulness)
- Calibrate against human experts

Reference: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Optional

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
)
from llama_index.llms.openai import OpenAI


# =============================================================================
# Model-Based Grader Classes
# =============================================================================

@dataclass
class ModelGraderResult:
    """Result from a model-based evaluation."""
    dimension: str
    score: float
    passed: bool
    feedback: str
    threshold: float


class CorrectnessGrader:
    """
    Evaluates if the agent's response is factually correct.

    Uses an LLM judge to compare the response against ground truth,
    accounting for paraphrasing and different ways of expressing the same facts.

    Rubric:
    - 5: Perfect match, all facts correct
    - 4: Minor omissions, core facts correct
    - 3: Partially correct, some errors
    - 2: Significant errors or missing information
    - 1: Mostly incorrect
    - 0: Completely wrong or irrelevant
    """

    def __init__(self, model: str = "gpt-4o", threshold: float = 3.5):
        self.llm = OpenAI(model=model, temperature=0)
        self.evaluator = CorrectnessEvaluator(llm=self.llm)
        self.threshold = threshold

    async def grade(
        self,
        query: str,
        response: str,
        ground_truth: str
    ) -> ModelGraderResult:
        """Evaluate correctness of response against ground truth."""
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                reference=ground_truth,
            )

            score = result.score if result.score is not None else 0
            return ModelGraderResult(
                dimension="correctness",
                score=score,
                passed=score >= self.threshold,
                feedback=result.feedback or "No feedback provided",
                threshold=self.threshold,
            )
        except Exception as e:
            return ModelGraderResult(
                dimension="correctness",
                score=0,
                passed=False,
                feedback=f"Evaluation error: {str(e)}",
                threshold=self.threshold,
            )


class RelevancyGrader:
    """
    Evaluates if retrieved context is relevant to the query.

    Uses an LLM judge to assess whether the contexts retrieved
    by the agent actually help answer the question.

    Rubric:
    - 1.0: Highly relevant, context directly answers query
    - 0.5: Partially relevant, some useful information
    - 0.0: Not relevant, context doesn't help
    """

    def __init__(self, model: str = "gpt-4o", threshold: float = 0.5):
        self.llm = OpenAI(model=model, temperature=0)
        self.evaluator = RelevancyEvaluator(llm=self.llm)
        self.threshold = threshold

    async def grade(
        self,
        query: str,
        response: str,
        contexts: list[str]
    ) -> ModelGraderResult:
        """Evaluate relevancy of retrieved contexts."""
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
            )

            score = result.score if result.score is not None else 0
            return ModelGraderResult(
                dimension="relevancy",
                score=score,
                passed=score >= self.threshold,
                feedback=result.feedback or "No feedback provided",
                threshold=self.threshold,
            )
        except Exception as e:
            return ModelGraderResult(
                dimension="relevancy",
                score=0,
                passed=False,
                feedback=f"Evaluation error: {str(e)}",
                threshold=self.threshold,
            )


class FaithfulnessGrader:
    """
    Evaluates if the response is grounded in the retrieved context.

    Uses an LLM judge to verify the agent isn't hallucinating -
    every claim in the response should be supported by the context.

    Rubric:
    - 1.0: Fully grounded, all claims supported
    - 0.5: Partially grounded, some unsupported claims
    - 0.0: Not grounded, significant hallucination
    """

    def __init__(self, model: str = "gpt-4o", threshold: float = 0.5):
        self.llm = OpenAI(model=model, temperature=0)
        self.evaluator = FaithfulnessEvaluator(llm=self.llm)
        self.threshold = threshold

    async def grade(
        self,
        query: str,
        response: str,
        contexts: list[str]
    ) -> ModelGraderResult:
        """Evaluate faithfulness of response to context."""
        try:
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
            )

            score = result.score if result.score is not None else 0
            return ModelGraderResult(
                dimension="faithfulness",
                score=score,
                passed=score >= self.threshold,
                feedback=result.feedback or "No feedback provided",
                threshold=self.threshold,
            )
        except Exception as e:
            return ModelGraderResult(
                dimension="faithfulness",
                score=0,
                passed=False,
                feedback=f"Evaluation error: {str(e)}",
                threshold=self.threshold,
            )


class CustomRubricGrader:
    """
    Custom model-based grader with user-defined rubric.

    Allows defining specific evaluation criteria for domain-specific
    quality dimensions beyond the standard metrics.
    """

    def __init__(
        self,
        dimension: str,
        rubric: str,
        model: str = "gpt-4o",
        threshold: float = 3.0
    ):
        self.dimension = dimension
        self.rubric = rubric
        self.llm = OpenAI(model=model, temperature=0)
        self.threshold = threshold

    async def grade(
        self,
        query: str,
        response: str,
        ground_truth: Optional[str] = None
    ) -> ModelGraderResult:
        """Evaluate using custom rubric."""

        prompt = f"""You are evaluating an AI agent's response.

EVALUATION DIMENSION: {self.dimension}

RUBRIC:
{self.rubric}

USER QUERY:
{query}

AGENT RESPONSE:
{response}

{"EXPECTED ANSWER:" + chr(10) + ground_truth if ground_truth else ""}

Based on the rubric, provide:
1. A score from 0-5 (where 5 is best)
2. Brief feedback explaining your score

Respond in format:
SCORE: [number]
FEEDBACK: [your feedback]
"""

        try:
            result = await self.llm.acomplete(prompt)
            text = result.text

            # Parse score
            import re
            score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', text)
            score = float(score_match.group(1)) if score_match else 0

            # Parse feedback
            feedback_match = re.search(r'FEEDBACK:\s*(.+)', text, re.DOTALL)
            feedback = feedback_match.group(1).strip() if feedback_match else text

            return ModelGraderResult(
                dimension=self.dimension,
                score=score,
                passed=score >= self.threshold,
                feedback=feedback,
                threshold=self.threshold,
            )
        except Exception as e:
            return ModelGraderResult(
                dimension=self.dimension,
                score=0,
                passed=False,
                feedback=f"Evaluation error: {str(e)}",
                threshold=self.threshold,
            )


# =============================================================================
# Combined Model-Based Evaluator
# =============================================================================

class ModelBasedEvaluator:
    """
    Combined evaluator that runs all model-based graders.

    Following Anthropic's recommendation to use isolated judges for
    each dimension to avoid conflation of different quality aspects.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.correctness = CorrectnessGrader(model=model, threshold=3.5)
        self.relevancy = RelevancyGrader(model=model, threshold=0.5)
        self.faithfulness = FaithfulnessGrader(model=model, threshold=0.5)

    async def evaluate_all(
        self,
        query: str,
        response: str,
        ground_truth: str,
        contexts: list[str]
    ) -> dict[str, ModelGraderResult]:
        """Run all model-based evaluations."""

        results = {}

        # Run evaluations (could be parallelized)
        results["correctness"] = await self.correctness.grade(
            query, response, ground_truth
        )

        results["relevancy"] = await self.relevancy.grade(
            query, response, contexts
        )

        results["faithfulness"] = await self.faithfulness.grade(
            query, response, contexts
        )

        return results


# =============================================================================
# Helper Functions
# =============================================================================

async def run_agent_and_get_response(agent, query: str) -> tuple[str, list[str]]:
    """Run agent and return response with contexts."""
    contexts = []

    handler = agent.run(user_msg=query)

    async for event in handler.stream_events():
        # Capture any context from tool outputs
        if hasattr(event, 'tool_output'):
            output = str(event.tool_output)
            if len(output) > 50:  # Likely contains context
                contexts.append(output)

    response = await handler
    response_text = str(response)

    # If no contexts captured, use response as context
    if not contexts:
        contexts = [response_text]

    return response_text, contexts


# =============================================================================
# Tests: Correctness Evaluation
# =============================================================================

class TestCorrectnessGrader:
    """Tests for model-based correctness evaluation."""

    @pytest.mark.asyncio
    async def test_needle_correctness(self, initialized_agent, needle_cases):
        """Evaluate correctness of needle query responses."""
        grader = CorrectnessGrader(model="gpt-4o", threshold=3.5)
        test_case = needle_cases[0]

        response, _ = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(
            query=test_case.query,
            response=response,
            ground_truth=test_case.ground_truth,
        )

        print("\n" + "="*60)
        print("CORRECTNESS EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"Ground Truth: {test_case.ground_truth}")
        print(f"Response: {response[:300]}...")
        print(f"\nScore: {result.score}/5.0 (threshold: {result.threshold})")
        print(f"Passed: {result.passed}")
        print(f"Feedback: {result.feedback}")
        print("="*60)

        assert result.passed, f"Correctness below threshold: {result.feedback}"

    @pytest.mark.asyncio
    async def test_summary_correctness(self, initialized_agent, summary_cases):
        """Evaluate correctness of summary query responses."""
        grader = CorrectnessGrader(model="gpt-4o", threshold=3.0)
        test_case = summary_cases[0]

        response, _ = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(
            query=test_case.query,
            response=response,
            ground_truth=test_case.ground_truth,
        )

        print("\n" + "="*60)
        print("SUMMARY CORRECTNESS EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"Response: {response[:400]}...")
        print(f"\nScore: {result.score}/5.0")
        print(f"Feedback: {result.feedback}")
        print("="*60)

        assert result.passed, f"Summary correctness below threshold: {result.feedback}"


# =============================================================================
# Tests: Relevancy Evaluation
# =============================================================================

class TestRelevancyGrader:
    """Tests for model-based relevancy evaluation."""

    @pytest.mark.asyncio
    async def test_needle_relevancy(self, initialized_agent, needle_cases):
        """Evaluate relevancy of retrieved context for needle queries."""
        grader = RelevancyGrader(model="gpt-4o", threshold=0.5)
        test_case = needle_cases[0]

        response, contexts = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(
            query=test_case.query,
            response=response,
            contexts=contexts,
        )

        print("\n" + "="*60)
        print("RELEVANCY EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"Contexts captured: {len(contexts)}")
        print(f"\nScore: {result.score}/1.0 (threshold: {result.threshold})")
        print(f"Passed: {result.passed}")
        print(f"Feedback: {result.feedback}")
        print("="*60)

        assert result.passed, f"Relevancy below threshold: {result.feedback}"


# =============================================================================
# Tests: Faithfulness Evaluation
# =============================================================================

class TestFaithfulnessGrader:
    """Tests for model-based faithfulness evaluation."""

    @pytest.mark.asyncio
    async def test_needle_faithfulness(self, initialized_agent, needle_cases):
        """Evaluate faithfulness - response grounded in context."""
        grader = FaithfulnessGrader(model="gpt-4o", threshold=0.5)
        test_case = needle_cases[0]

        response, contexts = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(
            query=test_case.query,
            response=response,
            contexts=contexts,
        )

        print("\n" + "="*60)
        print("FAITHFULNESS EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"Response: {response[:300]}...")
        print(f"\nScore: {result.score}/1.0 (threshold: {result.threshold})")
        print(f"Passed: {result.passed}")
        print(f"Feedback: {result.feedback}")
        print("="*60)

        assert result.passed, f"Faithfulness below threshold: {result.feedback}"


# =============================================================================
# Tests: Custom Rubric Evaluation
# =============================================================================

class TestCustomRubricGrader:
    """Tests for custom rubric-based evaluation."""

    @pytest.mark.asyncio
    async def test_professional_tone(self, initialized_agent, needle_cases):
        """Evaluate if response maintains professional insurance tone."""
        rubric = """
        Evaluate the professional quality of the response for insurance claim analysis:
        5 - Highly professional, uses appropriate terminology, clear and precise
        4 - Professional tone, minor informal language
        3 - Adequate, but could be more professional
        2 - Somewhat informal or imprecise
        1 - Unprofessional or overly casual
        0 - Completely inappropriate tone
        """

        grader = CustomRubricGrader(
            dimension="professional_tone",
            rubric=rubric,
            model="gpt-4o",
            threshold=3.0
        )

        test_case = needle_cases[0]
        response, _ = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        result = await grader.grade(
            query=test_case.query,
            response=response,
        )

        print("\n" + "="*60)
        print("CUSTOM RUBRIC: PROFESSIONAL TONE")
        print("="*60)
        print(f"Response: {response[:300]}...")
        print(f"\nScore: {result.score}/5.0")
        print(f"Feedback: {result.feedback}")
        print("="*60)

        assert result.passed, f"Professional tone below threshold: {result.feedback}"


# =============================================================================
# Tests: Combined Model-Based Evaluation
# =============================================================================

class TestCombinedModelBasedEvaluation:
    """Tests combining all model-based graders."""

    @pytest.mark.asyncio
    async def test_full_model_evaluation(self, initialized_agent, needle_cases):
        """Run complete model-based evaluation suite."""
        evaluator = ModelBasedEvaluator(model="gpt-4o")
        test_case = needle_cases[0]

        response, contexts = await run_agent_and_get_response(
            initialized_agent,
            test_case.query
        )

        results = await evaluator.evaluate_all(
            query=test_case.query,
            response=response,
            ground_truth=test_case.ground_truth,
            contexts=contexts,
        )

        print("\n" + "="*60)
        print("FULL MODEL-BASED EVALUATION")
        print("="*60)
        print(f"Query: {test_case.query}")
        print(f"Response: {response[:300]}...")

        all_passed = True
        for dimension, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"\n--- {dimension.upper()} [{status}] ---")
            print(f"  Score: {result.score} (threshold: {result.threshold})")
            print(f"  Feedback: {result.feedback[:200]}...")
            if not result.passed:
                all_passed = False

        print("\n" + "="*60)
        print(f"OVERALL: {'PASSED' if all_passed else 'FAILED'}")
        print("="*60)

        assert all_passed, "One or more model-based evaluations failed"

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, initialized_agent, test_cases):
        """Batch evaluate multiple test cases."""
        evaluator = ModelBasedEvaluator(model="gpt-4o")

        results_summary = []

        # Evaluate first 3 test cases (to keep runtime reasonable)
        for test_case in test_cases[:3]:
            response, contexts = await run_agent_and_get_response(
                initialized_agent,
                test_case.query
            )

            results = await evaluator.evaluate_all(
                query=test_case.query,
                response=response,
                ground_truth=test_case.ground_truth,
                contexts=contexts,
            )

            summary = {
                "query": test_case.query[:50] + "...",
                "category": test_case.category,
                "correctness": results["correctness"].score,
                "relevancy": results["relevancy"].score,
                "faithfulness": results["faithfulness"].score,
            }
            results_summary.append(summary)

        print("\n" + "="*70)
        print("BATCH EVALUATION SUMMARY")
        print("="*70)
        print(f"{'Query':<55} {'Cat':<8} {'Corr':>6} {'Rel':>6} {'Faith':>6}")
        print("-"*70)

        for r in results_summary:
            print(f"{r['query']:<55} {r['category']:<8} {r['correctness']:>6.2f} {r['relevancy']:>6.2f} {r['faithfulness']:>6.2f}")

        # Calculate averages
        avg_corr = sum(r["correctness"] for r in results_summary) / len(results_summary)
        avg_rel = sum(r["relevancy"] for r in results_summary) / len(results_summary)
        avg_faith = sum(r["faithfulness"] for r in results_summary) / len(results_summary)

        print("-"*70)
        print(f"{'AVERAGES':<55} {'':<8} {avg_corr:>6.2f} {avg_rel:>6.2f} {avg_faith:>6.2f}")
        print("="*70)


# =============================================================================
# Run standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
