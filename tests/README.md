# Agent Evaluation Test Suite

This test suite demonstrates three evaluation methods for agentic AI systems, based on [Anthropic's guide to demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Overview

| Grader Type | File | Purpose | Speed | Cost |
|-------------|------|---------|-------|------|
| **Code-Based** | `test_code_based.py` | Deterministic checks (tool calls, format, content) | Fast | Free |
| **Model-Based** | `test_model_based.py` | LLM-as-judge evaluation (correctness, relevancy, faithfulness) | Slow | $$$ |
| **Human-Based** | `test_human_based.py` | CLI framework for human evaluation | Manual | Time |

## Quick Start

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest tests/ -v -s

# Run specific grader type
uv run pytest tests/test_code_based.py -v -s
uv run pytest tests/test_model_based.py -v -s

# Run human evaluation CLI
uv run python tests/test_human_based.py
```

## Grader Details

### 1. Code-Based Graders (`test_code_based.py`)

Fast, cheap, and reproducible. Use for:
- **Tool call validation**: Verify correct tools are called in correct order
- **Format validation**: Check response contains expected patterns (dates, currency, IDs)
- **Content validation**: Verify key facts appear in response

```python
# Example: Tool Call Grader
grader = ToolCallGrader(strict_order=True)
result = grader.grade(
    actual_tools=["resolve_claim", "needle_expert"],
    expected_tools=["resolve_claim", "needle_expert"]
)
assert result.passed
```

### 2. Model-Based Graders (`test_model_based.py`)

Handle nuance and open-ended evaluation using LLM judges:
- **CorrectnessGrader**: Does answer match ground truth? (0-5 scale)
- **RelevancyGrader**: Is retrieved context relevant? (0-1 scale)
- **FaithfulnessGrader**: Is answer grounded in context? (0-1 scale)
- **CustomRubricGrader**: Evaluate with custom criteria

```python
# Example: Correctness Evaluation
grader = CorrectnessGrader(model="gpt-4o", threshold=3.5)
result = await grader.grade(
    query="What was the accident time?",
    response="The accident occurred at 11:30 PM",
    ground_truth="October 14, 2023, at 11:30 PM"
)
```

### 3. Human-Based Graders (`test_human_based.py`)

Gold standard for calibration and subjective evaluation:
- Interactive CLI for scoring responses
- Predefined rubrics for consistency
- Session management and result persistence
- Calibration tools for model-human agreement

```bash
# Run human evaluation session
uv run python tests/test_human_based.py
```

## Test Cases

The test suite includes 7 test cases across 3 categories:
- **Needle** (3): Specific fact retrieval questions
- **Summary** (2): Broad comprehension questions
- **MCP** (2): Policy validation tool usage

## GitHub Actions

Tests can be run manually via GitHub Actions:

1. Go to Actions → "Agent Evaluation Tests"
2. Click "Run workflow"
3. Select test type (all, code-based, or model-based)
4. Results are uploaded as artifacts

**Note**: Requires `OPENAI_API_KEY` secret configured in repository settings.

## Directory Structure

```
tests/
├── __init__.py           # Package marker
├── conftest.py           # Shared fixtures and test cases
├── test_code_based.py    # Code-based grader tests
├── test_model_based.py   # Model-based grader tests
├── test_human_based.py   # Human evaluation CLI
└── README.md             # This file
```

## Configuration

- `pytest.ini`: Test configuration and markers
- `.github/workflows/agent-evals.yml`: CI/CD workflow

## Best Practices

Following Anthropic's recommendations:

1. **Grade outcomes, not paths**: Check what the agent produced, not exactly how it got there
2. **Isolated judges**: Use separate evaluators for each dimension
3. **Start small**: 20-50 test cases drawn from real failures
4. **Calibrate regularly**: Use human evaluation to calibrate model judges
