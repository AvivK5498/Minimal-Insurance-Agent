"""
Shared fixtures and configuration for agent evaluation tests.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# =============================================================================
# Test Case Data Structure
# =============================================================================

@dataclass
class EvalTestCase:
    """A single test case for evaluation."""
    query: str
    ground_truth: str
    category: str  # "needle", "summary", or "mcp"
    claim_id: str
    expected_tools: list[str] = None  # Expected tool calls for code-based grading

    def __post_init__(self):
        if self.expected_tools is None:
            self.expected_tools = []


# =============================================================================
# Test Cases Dataset (40 comprehensive test cases)
# =============================================================================

TEST_CASES = [
    # =========================================================================
    # CLM-89921 (Multi-Vehicle Collision) - NEEDLE QUESTIONS
    # =========================================================================
    EvalTestCase(
        query="For claim CLM-89921, what was the exact accident time according to the police report?",
        ground_truth="October 14, 2023, at 11:30 PM",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what is the policy holder's address?",
        ground_truth="1189 Silverwood Lane, San Mateo, CA",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what was the insured vehicle's speed at the time of the incident?",
        ground_truth="approximately 45 mph",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, who was the reporting police officer?",
        ground_truth="Sgt. Daniel H. Ruiz, Badge 7713",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what is the collision deductible amount?",
        ground_truth="$1,000",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what was the total labor cost in the repair estimate?",
        ground_truth="$2,024.00 (22 hours at $92/hr)",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what was the hazmat towing fee?",
        ground_truth="$452.50",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what hospital was Sarah Connor transported to?",
        ground_truth="Stanford Hospital Emergency Department",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what exclusion code might apply to this claim?",
        ground_truth="Exclusion Code EX-9982 (unauthorized commercial use)",
        category="needle",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "needle_expert"],
    ),

    # =========================================================================
    # CLM-44217-PD (Water Damage) - NEEDLE QUESTIONS
    # =========================================================================
    EvalTestCase(
        query="For claim CLM-44217-PD, what specific type of pipe ruptured behind the laundry room wall?",
        ground_truth="PEX line",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what is Jonathan Reyes's occupation?",
        ground_truth="Fire Department Logistics Coordinator",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, how many square feet of flooring was saturated?",
        ground_truth="approximately 380 sq ft",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what was the total ALE (Additional Living Expenses) claimed?",
        ground_truth="$1,468.92",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what is the policy deductible?",
        ground_truth="$2,000 All-Perils",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what time did the humidity rise begin according to thermostat logs?",
        ground_truth="around 2:15 AM",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what mitigation company responded to the incident?",
        ground_truth="EverDry Response Unit",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what was the cost per square foot for laminate flooring replacement?",
        ground_truth="$5.70 per sq ft",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what year was the home built?",
        ground_truth="1983",
        category="needle",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "needle_expert"],
    ),

    # =========================================================================
    # CLM-77182-CM (Theft/Equipment) - NEEDLE QUESTIONS
    # =========================================================================
    EvalTestCase(
        query="For claim CLM-77182-CM, what was the total estimated replacement cost of equipment before depreciation?",
        ground_truth="$22,515.00",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, what is the serial number of the stolen RED camera?",
        ground_truth="KMD002918",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, what was the policy deductible?",
        ground_truth="$5,000 per occurrence",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, what type of vehicle was seen near the loading zone?",
        ground_truth="mid-2000s silver Nissan Frontier",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, what time was the suspicious vehicle observed?",
        ground_truth="around 5:51 AM",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, what was the emergency rental cost for replacement gear?",
        ground_truth="$1,180",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, who is the primary contact for the business?",
        ground_truth="Michael Torres, Creative Director",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, what was the total depreciation applied?",
        ground_truth="$4,780.00",
        category="needle",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "needle_expert"],
    ),

    # =========================================================================
    # SUMMARY QUESTIONS
    # =========================================================================
    EvalTestCase(
        query="For claim CLM-89921, what was the general sequence of events in the multi-vehicle collision?",
        ground_truth="Sarah Connor's vehicle was traveling on Highway 101 when she observed sudden brake lights from a black SUV ahead. She braked hard but made contact with the SUV's rear bumper. The SUV swerved into an adjacent lane striking a Honda Civic, and a delivery van then collided into the insured's Camry from behind.",
        category="summary",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "summary_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, summarize the medical treatment Sarah Connor received.",
        ground_truth="Sarah Connor was transported to Stanford Hospital with chest tightness from airbag deployment, clavicle bruising from seatbelt, and minor concussion symptoms. She received CT scan, X-ray, ECG, analgesics, ice compression, and was observed for 4 hours before discharge with concussion protocols.",
        category="summary",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "summary_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-89921, what are the main red flags identified by the adjuster?",
        ground_truth="Timeline inconsistency between crash time and hospital admission, driver identity ambiguity between Sarah Connor and Sarah O'Connor-Smith, potential commercial usage of personal vehicle violating exclusion EX-9982, and witness familiarity concerns.",
        category="summary",
        claim_id="CLM-89921",
        expected_tools=["resolve_claim", "summary_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, what caused the water intrusion and what major areas of the home were affected?",
        ground_truth="A PEX line leading to the washing machine intake valve ruptured. Major areas affected include approximately 380 sq ft of laminate flooring, laundry room, guest bedroom wall, subfloor spaces, and crawl area. Mold bloom was observed along the wall plate.",
        category="summary",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "summary_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-44217-PD, describe the mitigation efforts taken.",
        ground_truth="The insured shut off water supply, used towels and wet vac temporarily. EverDry Response Unit arrived same day, installed dehumidifiers and air scrubbers, set up containment barriers, opened wall sections for airflow, removed saturated insulation, and applied mold treatment.",
        category="summary",
        claim_id="CLM-44217-PD",
        expected_tools=["resolve_claim", "summary_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, summarize the theft incident and what was stolen.",
        ground_truth="On July 18, 2024, intruders accessed Lighthouse Creative Media through the rear loading entrance using a pry-bar. Stolen items included a RED Komodo 6K camera, Canon RF 24-70mm lens, DJI Ronin gimbal, MacBook Pro, Sennheiser wireless audio kit, and a 2TB SSD with client assets.",
        category="summary",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "summary_expert"],
    ),
    EvalTestCase(
        query="For claim CLM-77182-CM, what business operations were impacted by the theft?",
        ground_truth="Three projects were impacted: a commercial shoot for Pacific Coast Coffee Roasters was delayed 5 days, a wedding videography package required same-day rental equipment, and an agency project for Monterey Travel Bureau had editing delayed due to missing SSD files.",
        category="summary",
        claim_id="CLM-77182-CM",
        expected_tools=["resolve_claim", "summary_expert"],
    ),

    # =========================================================================
    # MCP / POLICY VALIDATION QUESTIONS
    # =========================================================================
    EvalTestCase(
        query="Use the policy validation tool to check: Is $9,766.90 within a $100,000 policy limit?",
        ground_truth="Yes, the claim is within policy limits. Risk level: Low (9.77% of limit).",
        category="mcp",
        claim_id="CLM-89921",
        expected_tools=["validate_policy_limit"],
    ),
    EvalTestCase(
        query="Run the policy validation tool: Is a claim of $150,000 within a $100,000 property damage limit?",
        ground_truth="No, the claim EXCEEDS policy limits by $50,000 (150% of limit).",
        category="mcp",
        claim_id="CLM-89921",
        expected_tools=["validate_policy_limit"],
    ),
    EvalTestCase(
        query="Use the policy validation tool: Is $6,692 within a $450,000 dwelling coverage limit?",
        ground_truth="Yes, the claim is within policy limits. Risk level: Low (1.49% of limit).",
        category="mcp",
        claim_id="CLM-44217-PD",
        expected_tools=["validate_policy_limit"],
    ),
    EvalTestCase(
        query="Validate policy: Is $22,515 within a $95,000 equipment floater limit?",
        ground_truth="Yes, the claim is within policy limits. Risk level: Low (23.7% of limit).",
        category="mcp",
        claim_id="CLM-77182-CM",
        expected_tools=["validate_policy_limit"],
    ),
    EvalTestCase(
        query="Use policy validation tool: Is $85,000 within a $100,000 bodily injury limit?",
        ground_truth="Yes, but Risk level: High (85% of limit). Close to policy maximum.",
        category="mcp",
        claim_id="CLM-89921",
        expected_tools=["validate_policy_limit"],
    ),
    EvalTestCase(
        query="Run policy validation: Is $50,000 within a $60,000 loss of use coverage?",
        ground_truth="Yes, but Risk level: High (83.3% of limit). Close to policy maximum.",
        category="mcp",
        claim_id="CLM-44217-PD",
        expected_tools=["validate_policy_limit"],
    ),
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def llm_settings():
    """Configure LlamaIndex settings."""
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    return Settings


@pytest.fixture(scope="session")
def initialized_agent(llm_settings):
    """Initialize the agent system once for all tests."""
    from main import initialize_system
    agent, documents = initialize_system()
    return agent


@pytest.fixture
def test_cases():
    """Provide test cases for evaluation."""
    return TEST_CASES


@pytest.fixture
def needle_cases():
    """Provide only needle test cases."""
    return [tc for tc in TEST_CASES if tc.category == "needle"]


@pytest.fixture
def summary_cases():
    """Provide only summary test cases."""
    return [tc for tc in TEST_CASES if tc.category == "summary"]


@pytest.fixture
def mcp_cases():
    """Provide only MCP test cases."""
    return [tc for tc in TEST_CASES if tc.category == "mcp"]
