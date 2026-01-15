"""
Test Results Analyzer

Parses pytest output and uses OpenAI to generate a summary table of test results.
"""

import os
import sys
import re
from openai import OpenAI


def parse_pytest_output(output: str) -> dict:
    """Parse pytest output to extract test results."""
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "test_details": [],
        "raw_output": output[-15000:] if len(output) > 15000 else output,  # Limit size
    }

    # Parse summary line (e.g., "5 passed, 2 failed, 1 error in 10.5s")
    summary_match = re.search(
        r'=+ (?:short test summary info|FAILURES)? ?=+.*?(\d+) passed',
        output, re.DOTALL
    )
    if summary_match:
        results["passed"] = int(summary_match.group(1))

    # Alternative summary parsing
    final_summary = re.search(
        r'=+ ([\d\w\s,]+) in [\d.]+s =+',
        output
    )
    if final_summary:
        summary_text = final_summary.group(1)

        passed_match = re.search(r'(\d+) passed', summary_text)
        if passed_match:
            results["passed"] = int(passed_match.group(1))

        failed_match = re.search(r'(\d+) failed', summary_text)
        if failed_match:
            results["failed"] = int(failed_match.group(1))

        error_match = re.search(r'(\d+) error', summary_text)
        if error_match:
            results["errors"] = int(error_match.group(1))

        skipped_match = re.search(r'(\d+) skipped', summary_text)
        if skipped_match:
            results["skipped"] = int(skipped_match.group(1))

    results["total_tests"] = (
        results["passed"] + results["failed"] +
        results["errors"] + results["skipped"]
    )

    # Extract individual test results
    test_pattern = re.compile(
        r'(PASSED|FAILED|ERROR)\s+(tests/[\w/]+\.py::[\w\[\]_-]+)'
    )
    for match in test_pattern.finditer(output):
        results["test_details"].append({
            "status": match.group(1),
            "test": match.group(2),
        })

    return results


def analyze_with_llm(results: dict, test_type: str) -> str:
    """Use OpenAI to analyze test results and generate summary table."""

    client = OpenAI()

    prompt = f"""Analyze these pytest results for an Insurance Claims RAG Agent evaluation system.

TEST TYPE: {test_type}

SUMMARY:
- Total Tests: {results['total_tests']}
- Passed: {results['passed']}
- Failed: {results['failed']}
- Errors: {results['errors']}
- Skipped: {results['skipped']}

RAW TEST OUTPUT (last portion):
```
{results['raw_output']}
```

Based on the test output, generate a summary table in this EXACT format. Analyze what categories of tests ran and their results:

| Category | Score | Status | Analysis |
|----------|-------|--------|----------|
| Tool Calling | X / Y Passed | SUCCESS/FAILED | [Brief analysis of tool call validation tests] |
| Response Format | X / Y Passed | SUCCESS/FAILED | [Brief analysis of format validation tests] |
| Content Logic | X / Y Passed | SUCCESS/FAILED | [Brief analysis of content/logic tests] |
| Correctness | X.X / 5.0 (Avg) | SUCCESS/FAILED | [Brief analysis of correctness evaluations] |
| Faithfulness | X.X / 1.0 (Avg) | SUCCESS/FAILED | [Brief analysis of faithfulness/hallucination tests] |
| Relevancy | X.X / 1.0 (Avg) | SUCCESS/FAILED | [Brief analysis of relevancy tests] |

Rules:
1. Only include categories that have actual test results in the output
2. If a category has no tests, omit it from the table
3. Use SUCCESS if all tests in category passed, FAILED if any failed
4. Be specific about what failed if there are failures
5. Keep analysis brief (under 15 words per cell)
6. If scores aren't explicitly shown, estimate based on pass/fail counts

Return ONLY the markdown table, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a test results analyst. Generate concise summary tables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing results: {str(e)}"


def main():
    """Main entry point."""
    # Read test output from file or stdin
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        test_type = sys.argv[2] if len(sys.argv) > 2 else "Agent Evaluation"

        try:
            with open(output_file, 'r') as f:
                output = f.read()
        except FileNotFoundError:
            print(f"Error: File {output_file} not found")
            sys.exit(1)
    else:
        output = sys.stdin.read()
        test_type = "Agent Evaluation"

    if not output.strip():
        print("No test output to analyze")
        sys.exit(1)

    # Parse and analyze
    results = parse_pytest_output(output)

    print("\n" + "="*70)
    print("📊 TEST RESULTS ANALYSIS")
    print("="*70 + "\n")

    # Generate LLM analysis
    analysis = analyze_with_llm(results, test_type)
    print(analysis)

    print("\n" + "="*70)
    print(f"Total: {results['passed']}/{results['total_tests']} tests passed")
    print("="*70 + "\n")

    # Return exit code based on results
    if results['failed'] > 0 or results['errors'] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
