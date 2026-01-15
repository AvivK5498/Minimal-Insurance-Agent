"""
Human-Based Graders for Agent Evaluation

Human graders set the gold standard but don't scale. They are best used for:
- Calibrating model-based graders
- Evaluating subjective quality (tone, helpfulness)
- Spot-checking production outputs
- Creating ground truth datasets

This module provides a CLI framework for human evaluation sessions.

Reference: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HumanGraderResult:
    """Result from a human evaluation."""
    query: str
    response: str
    ground_truth: str
    dimension: str
    score: float
    feedback: str
    grader_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HumanEvaluationSession:
    """A complete human evaluation session."""
    session_id: str
    grader_id: str
    started_at: str
    completed_at: Optional[str] = None
    results: list[HumanGraderResult] = field(default_factory=list)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "grader_id": self.grader_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [asdict(r) for r in self.results],
        }


# =============================================================================
# CLI Colors and Formatting
# =============================================================================

class Colors:
    """Terminal colors for CLI output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_section(title: str, content: str, color: str = Colors.CYAN):
    """Print a labeled section."""
    print(f"{color}{Colors.BOLD}{title}:{Colors.ENDC}")
    print(f"  {content}\n")


def print_rubric(rubric: dict):
    """Print evaluation rubric."""
    print(f"{Colors.YELLOW}{Colors.BOLD}SCORING RUBRIC:{Colors.ENDC}")
    for score, description in sorted(rubric.items(), reverse=True):
        print(f"  {Colors.GREEN}{score}{Colors.ENDC}: {description}")
    print()


# =============================================================================
# Rubrics for Human Evaluation
# =============================================================================

RUBRICS = {
    "correctness": {
        5: "Perfect - All facts correct, complete answer",
        4: "Good - Minor omissions, core facts correct",
        3: "Adequate - Partially correct, some errors",
        2: "Poor - Significant errors or missing info",
        1: "Bad - Mostly incorrect",
        0: "Wrong - Completely incorrect or irrelevant",
    },
    "relevancy": {
        5: "Highly relevant - Directly answers the question",
        4: "Relevant - Answers question with minor tangents",
        3: "Somewhat relevant - Partial answer, some off-topic",
        2: "Marginally relevant - Mostly off-topic",
        1: "Barely relevant - Almost entirely off-topic",
        0: "Not relevant - Does not address question",
    },
    "faithfulness": {
        5: "Fully grounded - All claims supported by context",
        4: "Mostly grounded - Minor unsupported details",
        3: "Partially grounded - Some hallucination",
        2: "Weakly grounded - Significant hallucination",
        1: "Barely grounded - Mostly hallucinated",
        0: "Not grounded - Pure hallucination",
    },
    "helpfulness": {
        5: "Extremely helpful - Exceeds expectations",
        4: "Very helpful - Meets all needs",
        3: "Helpful - Meets basic needs",
        2: "Somewhat helpful - Partially useful",
        1: "Barely helpful - Minimal utility",
        0: "Not helpful - No value provided",
    },
    "tone": {
        5: "Perfect - Professional, clear, appropriate",
        4: "Good - Professional with minor issues",
        3: "Acceptable - Adequate professionalism",
        2: "Below average - Some inappropriate language",
        1: "Poor - Unprofessional",
        0: "Unacceptable - Completely inappropriate",
    },
}


# =============================================================================
# Human Evaluation CLI
# =============================================================================

class HumanEvaluationCLI:
    """
    CLI framework for human evaluation of agent responses.

    Usage:
        cli = HumanEvaluationCLI()
        await cli.run_session(agent, test_cases, dimensions=["correctness", "helpfulness"])
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "human_eval_results"
        self.output_dir.mkdir(exist_ok=True)

    def get_grader_id(self) -> str:
        """Prompt for grader identification."""
        print_header("HUMAN EVALUATION SESSION")
        print("Welcome to the human evaluation framework.")
        print("You will be shown agent responses and asked to score them.\n")

        grader_id = input(f"{Colors.CYAN}Enter your name/ID: {Colors.ENDC}").strip()
        if not grader_id:
            grader_id = "anonymous"
        return grader_id

    def get_score(self, dimension: str) -> tuple[float, str]:
        """Prompt human for a score and feedback."""
        rubric = RUBRICS.get(dimension, RUBRICS["correctness"])
        print_rubric(rubric)

        while True:
            try:
                score_input = input(f"{Colors.GREEN}Enter score (0-5): {Colors.ENDC}").strip()
                score = float(score_input)
                if 0 <= score <= 5:
                    break
                print(f"{Colors.RED}Score must be between 0 and 5.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Please enter a valid number.{Colors.ENDC}")

        feedback = input(f"{Colors.GREEN}Enter feedback (optional): {Colors.ENDC}").strip()
        return score, feedback

    async def evaluate_single(
        self,
        query: str,
        response: str,
        ground_truth: str,
        dimension: str,
        grader_id: str,
        index: int,
        total: int,
    ) -> HumanGraderResult:
        """Evaluate a single response."""

        print_header(f"EVALUATION {index}/{total}: {dimension.upper()}")

        print_section("QUERY", query, Colors.BLUE)
        print_section("GROUND TRUTH", ground_truth, Colors.YELLOW)
        print_section("AGENT RESPONSE", response[:500] + ("..." if len(response) > 500 else ""), Colors.CYAN)

        print(f"\n{Colors.BOLD}Please evaluate the {dimension.upper()} of this response:{Colors.ENDC}\n")

        score, feedback = self.get_score(dimension)

        return HumanGraderResult(
            query=query,
            response=response,
            ground_truth=ground_truth,
            dimension=dimension,
            score=score,
            feedback=feedback,
            grader_id=grader_id,
        )

    async def run_session(
        self,
        agent,
        test_cases: list,
        dimensions: list[str] = None,
        max_cases: int = None,
    ) -> HumanEvaluationSession:
        """
        Run a complete human evaluation session.

        Args:
            agent: The initialized agent to evaluate
            test_cases: List of test cases with query, ground_truth attributes
            dimensions: Which dimensions to evaluate (default: correctness, helpfulness)
            max_cases: Maximum number of cases to evaluate (for time constraints)
        """
        if dimensions is None:
            dimensions = ["correctness", "helpfulness"]

        grader_id = self.get_grader_id()

        session = HumanEvaluationSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            grader_id=grader_id,
            started_at=datetime.now().isoformat(),
        )

        cases_to_evaluate = test_cases[:max_cases] if max_cases else test_cases
        total_evals = len(cases_to_evaluate) * len(dimensions)

        print(f"\n{Colors.CYAN}Starting evaluation session with {len(cases_to_evaluate)} cases and {len(dimensions)} dimensions.{Colors.ENDC}")
        print(f"{Colors.CYAN}Total evaluations: {total_evals}{Colors.ENDC}")
        print(f"\n{Colors.YELLOW}Press Ctrl+C at any time to save and exit.{Colors.ENDC}\n")

        input(f"{Colors.GREEN}Press Enter to begin...{Colors.ENDC}")

        eval_count = 0
        try:
            for case in cases_to_evaluate:
                # Get agent response
                print(f"\n{Colors.YELLOW}Getting agent response...{Colors.ENDC}")
                handler = agent.run(user_msg=case.query)
                async for _ in handler.stream_events():
                    pass
                response = str(await handler)

                # Evaluate each dimension
                for dimension in dimensions:
                    eval_count += 1
                    result = await self.evaluate_single(
                        query=case.query,
                        response=response,
                        ground_truth=case.ground_truth,
                        dimension=dimension,
                        grader_id=grader_id,
                        index=eval_count,
                        total=total_evals,
                    )
                    session.results.append(result)

                    # Save progress after each evaluation
                    self._save_session(session)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Session interrupted. Saving progress...{Colors.ENDC}")

        session.completed_at = datetime.now().isoformat()
        self._save_session(session)

        self._print_summary(session)

        return session

    def _save_session(self, session: HumanEvaluationSession):
        """Save session to JSON file."""
        output_path = self.output_dir / f"human_eval_{session.session_id}.json"
        with open(output_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def _print_summary(self, session: HumanEvaluationSession):
        """Print evaluation summary."""
        print_header("EVALUATION SUMMARY")

        print(f"Session ID: {session.session_id}")
        print(f"Grader: {session.grader_id}")
        print(f"Completed: {len(session.results)} evaluations\n")

        if session.results:
            # Calculate averages by dimension
            dimension_scores = {}
            for result in session.results:
                if result.dimension not in dimension_scores:
                    dimension_scores[result.dimension] = []
                dimension_scores[result.dimension].append(result.score)

            print(f"{Colors.BOLD}Average Scores by Dimension:{Colors.ENDC}")
            for dim, scores in dimension_scores.items():
                avg = sum(scores) / len(scores)
                print(f"  {dim}: {avg:.2f}/5.0 ({len(scores)} evaluations)")

        output_path = self.output_dir / f"human_eval_{session.session_id}.json"
        print(f"\n{Colors.GREEN}Results saved to: {output_path}{Colors.ENDC}")


# =============================================================================
# Calibration Tool
# =============================================================================

class CalibrationTool:
    """
    Tool for calibrating model-based graders against human judgments.

    This is crucial for ensuring LLM judges align with human expectations.
    """

    def __init__(self, human_results_path: Path = None):
        self.human_results_dir = human_results_path or Path(__file__).parent.parent / "human_eval_results"

    def load_human_results(self) -> list[HumanGraderResult]:
        """Load all human evaluation results."""
        results = []
        if not self.human_results_dir.exists():
            return results

        for path in self.human_results_dir.glob("human_eval_*.json"):
            with open(path) as f:
                session_data = json.load(f)
                for r in session_data.get("results", []):
                    results.append(HumanGraderResult(**r))

        return results

    def calculate_agreement(
        self,
        human_scores: list[float],
        model_scores: list[float],
    ) -> dict:
        """Calculate agreement metrics between human and model scores."""
        if len(human_scores) != len(model_scores):
            raise ValueError("Score lists must have same length")

        n = len(human_scores)
        if n == 0:
            return {"error": "No scores to compare"}

        # Mean Absolute Error
        mae = sum(abs(h - m) for h, m in zip(human_scores, model_scores)) / n

        # Exact match rate (within 0.5)
        exact_matches = sum(1 for h, m in zip(human_scores, model_scores) if abs(h - m) <= 0.5)
        match_rate = exact_matches / n

        # Correlation (Pearson)
        mean_h = sum(human_scores) / n
        mean_m = sum(model_scores) / n

        cov = sum((h - mean_h) * (m - mean_m) for h, m in zip(human_scores, model_scores)) / n
        std_h = (sum((h - mean_h) ** 2 for h in human_scores) / n) ** 0.5
        std_m = (sum((m - mean_m) ** 2 for m in model_scores) / n) ** 0.5

        correlation = cov / (std_h * std_m) if std_h > 0 and std_m > 0 else 0

        return {
            "n_samples": n,
            "mean_absolute_error": mae,
            "exact_match_rate": match_rate,
            "correlation": correlation,
            "human_mean": mean_h,
            "model_mean": mean_m,
        }

    def print_calibration_report(self, dimension: str, metrics: dict):
        """Print a calibration report."""
        print_header(f"CALIBRATION REPORT: {dimension.upper()}")

        print(f"Samples compared: {metrics['n_samples']}")
        print(f"\n{Colors.BOLD}Agreement Metrics:{Colors.ENDC}")
        print(f"  Mean Absolute Error: {metrics['mean_absolute_error']:.3f}")
        print(f"  Exact Match Rate (±0.5): {metrics['exact_match_rate']:.1%}")
        print(f"  Correlation: {metrics['correlation']:.3f}")
        print(f"\n{Colors.BOLD}Score Means:{Colors.ENDC}")
        print(f"  Human Mean: {metrics['human_mean']:.2f}")
        print(f"  Model Mean: {metrics['model_mean']:.2f}")

        # Interpretation
        if metrics['mean_absolute_error'] < 0.5:
            quality = f"{Colors.GREEN}Excellent{Colors.ENDC}"
        elif metrics['mean_absolute_error'] < 1.0:
            quality = f"{Colors.YELLOW}Good{Colors.ENDC}"
        else:
            quality = f"{Colors.RED}Needs Improvement{Colors.ENDC}"

        print(f"\n{Colors.BOLD}Calibration Quality: {quality}")


# =============================================================================
# Pytest Integration
# =============================================================================

import pytest


class TestHumanEvaluationFramework:
    """Tests to verify the human evaluation framework works."""

    def test_rubrics_exist(self):
        """Verify all rubrics are defined."""
        required_dimensions = ["correctness", "relevancy", "faithfulness", "helpfulness", "tone"]
        for dim in required_dimensions:
            assert dim in RUBRICS, f"Missing rubric for {dim}"
            assert len(RUBRICS[dim]) == 6, f"Rubric for {dim} should have scores 0-5"

    def test_human_result_dataclass(self):
        """Verify HumanGraderResult works correctly."""
        result = HumanGraderResult(
            query="Test query",
            response="Test response",
            ground_truth="Expected answer",
            dimension="correctness",
            score=4.0,
            feedback="Good answer",
            grader_id="test_grader",
        )

        assert result.score == 4.0
        assert result.dimension == "correctness"
        assert result.timestamp is not None

    def test_session_serialization(self):
        """Verify session can be serialized to dict."""
        session = HumanEvaluationSession(
            session_id="test_123",
            grader_id="tester",
            started_at=datetime.now().isoformat(),
        )
        session.results.append(HumanGraderResult(
            query="Q",
            response="R",
            ground_truth="GT",
            dimension="correctness",
            score=5.0,
            feedback="Perfect",
            grader_id="tester",
        ))

        data = session.to_dict()
        assert data["session_id"] == "test_123"
        assert len(data["results"]) == 1

    def test_calibration_metrics(self):
        """Verify calibration calculation works."""
        tool = CalibrationTool()

        human_scores = [5.0, 4.0, 3.0, 4.0, 5.0]
        model_scores = [4.5, 4.0, 3.5, 3.5, 4.5]

        metrics = tool.calculate_agreement(human_scores, model_scores)

        assert metrics["n_samples"] == 5
        assert 0 <= metrics["mean_absolute_error"] <= 5
        assert 0 <= metrics["exact_match_rate"] <= 1
        assert -1 <= metrics["correlation"] <= 1


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Run human evaluation CLI."""
    from dotenv import load_dotenv
    load_dotenv()

    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from main import initialize_system
    from conftest import TEST_CASES

    # Configure settings
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Initialize agent
    print("Initializing system...")
    agent, _ = initialize_system()

    # Run human evaluation
    cli = HumanEvaluationCLI()
    await cli.run_session(
        agent=agent,
        test_cases=TEST_CASES,
        dimensions=["correctness", "helpfulness"],
        max_cases=3,  # Limit for demo
    )


if __name__ == "__main__":
    asyncio.run(main())
