"""
Stress Tests for Agent Evaluation

Tests concurrent requests, load handling, and performance under pressure.
"""

import asyncio
import time
import pytest
from dataclasses import dataclass
from typing import List


@dataclass
class StressTestResult:
    """Result from a stress test."""
    query: str
    response_time: float
    success: bool
    response_length: int
    error: str = None


async def timed_agent_query(agent, query: str) -> StressTestResult:
    """Run agent query and measure response time."""
    start = time.time()
    try:
        handler = agent.run(user_msg=query)
        response = await handler
        elapsed = time.time() - start
        return StressTestResult(
            query=query[:50],
            response_time=elapsed,
            success=True,
            response_length=len(str(response)),
        )
    except Exception as e:
        elapsed = time.time() - start
        return StressTestResult(
            query=query[:50],
            response_time=elapsed,
            success=False,
            response_length=0,
            error=str(e),
        )


class TestConcurrentRequests:
    """Test handling of concurrent requests."""

    @pytest.mark.asyncio
    async def test_sequential_queries(self, initialized_agent, test_cases):
        """Baseline: Run queries sequentially."""
        results = []

        for test_case in test_cases[:5]:
            result = await timed_agent_query(initialized_agent, test_case.query)
            results.append(result)

        print("\n" + "="*70)
        print("SEQUENTIAL QUERY RESULTS")
        print("="*70)
        print(f"{'Query':<50} {'Time':>8} {'Success':>8} {'Len':>6}")
        print("-"*70)

        total_time = 0
        for r in results:
            status = "OK" if r.success else "FAIL"
            print(f"{r.query:<50} {r.response_time:>7.2f}s {status:>8} {r.response_length:>6}")
            total_time += r.response_time

        print("-"*70)
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time: {total_time/len(results):.2f}s")

        # All should succeed
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_parallel_queries_small(self, initialized_agent, test_cases):
        """Test 3 concurrent queries."""
        queries = [tc.query for tc in test_cases[:3]]

        start = time.time()
        tasks = [timed_agent_query(initialized_agent, q) for q in queries]
        results = await asyncio.gather(*tasks)
        total_elapsed = time.time() - start

        print("\n" + "="*70)
        print("PARALLEL QUERY RESULTS (3 concurrent)")
        print("="*70)
        print(f"{'Query':<50} {'Time':>8} {'Success':>8}")
        print("-"*70)

        for r in results:
            status = "OK" if r.success else "FAIL"
            print(f"{r.query:<50} {r.response_time:>7.2f}s {status:>8}")

        print("-"*70)
        print(f"Wall clock time: {total_elapsed:.2f}s")
        print(f"Speedup vs sequential: {sum(r.response_time for r in results)/total_elapsed:.1f}x")

        success_rate = sum(1 for r in results if r.success) / len(results)
        print(f"Success rate: {success_rate*100:.0f}%")

        assert success_rate >= 0.66, "At least 2/3 should succeed"

    @pytest.mark.asyncio
    async def test_parallel_queries_medium(self, initialized_agent, test_cases):
        """Test 5 concurrent queries."""
        queries = [tc.query for tc in test_cases[:5]]

        start = time.time()
        tasks = [timed_agent_query(initialized_agent, q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_elapsed = time.time() - start

        # Handle both results and exceptions
        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed_results.append(StressTestResult(
                    query=queries[i][:50],
                    response_time=0,
                    success=False,
                    response_length=0,
                    error=str(r),
                ))
            else:
                processed_results.append(r)

        print("\n" + "="*70)
        print("PARALLEL QUERY RESULTS (5 concurrent)")
        print("="*70)

        success_count = sum(1 for r in processed_results if r.success)
        print(f"Successful: {success_count}/{len(processed_results)}")
        print(f"Wall clock time: {total_elapsed:.2f}s")

        # At least half should succeed under load
        assert success_count >= len(processed_results) // 2


class TestResponseTimeThresholds:
    """Test that responses meet time thresholds."""

    @pytest.mark.asyncio
    async def test_simple_query_response_time(self, initialized_agent, needle_cases):
        """Simple needle queries should respond within threshold."""
        test_case = needle_cases[0]
        threshold_seconds = 30  # Generous threshold for API variability

        result = await timed_agent_query(initialized_agent, test_case.query)

        print(f"\nQuery: {test_case.query[:50]}...")
        print(f"Response time: {result.response_time:.2f}s")
        print(f"Threshold: {threshold_seconds}s")

        assert result.success, f"Query failed: {result.error}"
        assert result.response_time < threshold_seconds, \
            f"Response too slow: {result.response_time:.2f}s > {threshold_seconds}s"

    @pytest.mark.asyncio
    async def test_complex_query_response_time(self, initialized_agent, summary_cases):
        """Summary queries may take longer but should still be bounded."""
        test_case = summary_cases[0]
        threshold_seconds = 45  # More generous for complex queries

        result = await timed_agent_query(initialized_agent, test_case.query)

        print(f"\nQuery: {test_case.query[:50]}...")
        print(f"Response time: {result.response_time:.2f}s")
        print(f"Threshold: {threshold_seconds}s")

        assert result.success, f"Query failed: {result.error}"
        assert result.response_time < threshold_seconds, \
            f"Response too slow: {result.response_time:.2f}s > {threshold_seconds}s"


class TestLoadPatterns:
    """Test different load patterns."""

    @pytest.mark.asyncio
    async def test_burst_then_pause(self, initialized_agent, test_cases):
        """Simulate burst of queries followed by pause."""
        burst_size = 3

        # Burst
        burst_queries = [tc.query for tc in test_cases[:burst_size]]
        start = time.time()
        tasks = [timed_agent_query(initialized_agent, q) for q in burst_queries]
        burst_results = await asyncio.gather(*tasks)
        burst_time = time.time() - start

        print("\n" + "="*70)
        print("BURST PATTERN TEST")
        print("="*70)
        print(f"Burst of {burst_size} queries: {burst_time:.2f}s")

        # Pause
        await asyncio.sleep(2)

        # Single query after pause
        single_result = await timed_agent_query(
            initialized_agent,
            test_cases[0].query
        )

        print(f"Single query after pause: {single_result.response_time:.2f}s")

        # System should still be responsive after burst
        assert single_result.success
        assert single_result.response_time < 30

    @pytest.mark.asyncio
    async def test_same_query_repeated(self, initialized_agent, needle_cases):
        """Test same query repeated multiple times."""
        test_case = needle_cases[0]
        repetitions = 3

        results = []
        for i in range(repetitions):
            result = await timed_agent_query(initialized_agent, test_case.query)
            results.append(result)

        print("\n" + "="*70)
        print("REPEATED QUERY TEST")
        print("="*70)
        print(f"Query: {test_case.query[:50]}...")

        for i, r in enumerate(results):
            status = "OK" if r.success else "FAIL"
            print(f"  Run {i+1}: {r.response_time:.2f}s [{status}]")

        times = [r.response_time for r in results if r.success]
        if times:
            print(f"\nAverage: {sum(times)/len(times):.2f}s")
            print(f"Min: {min(times):.2f}s")
            print(f"Max: {max(times):.2f}s")

        # All should succeed
        assert all(r.success for r in results)


class TestResourceUsage:
    """Test resource usage patterns."""

    @pytest.mark.asyncio
    async def test_response_size_consistency(self, initialized_agent, needle_cases):
        """Response sizes should be consistent for similar queries."""
        results = []

        for tc in needle_cases:
            result = await timed_agent_query(initialized_agent, tc.query)
            results.append(result)

        sizes = [r.response_length for r in results if r.success]

        print("\n" + "="*70)
        print("RESPONSE SIZE ANALYSIS")
        print("="*70)
        print(f"Response sizes: {sizes}")
        print(f"Average: {sum(sizes)/len(sizes):.0f} chars")
        print(f"Min: {min(sizes)} chars")
        print(f"Max: {max(sizes)} chars")

        # Responses should be reasonably sized (not empty, not huge)
        assert all(50 < s < 10000 for s in sizes), \
            "Response sizes should be reasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
