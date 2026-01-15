# Run with: uv run evaluation.py
"""
System Evaluation Script - LLM-as-a-Judge
Evaluates the Insurance Claim RAG system using Correctness, Relevancy, and Faithfulness metrics.
"""

import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from main import initialize_system

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

JUDGE_MODEL = "gpt-4o"
RESULTS_PATH = Path(__file__).parent / "evaluation_results.json"
CSV_PATH = Path(__file__).parent / "evaluation_results.csv"


# =============================================================================
# Test Dataset
# =============================================================================

@dataclass
class TestCase:
    """A single test case for evaluation."""
    query: str
    ground_truth: str
    category: str  # "needle", "summary", or "mcp"
    claim_id: str


# Test cases covering all three claims and query types
TEST_CASES = [
    # ==========================================================================
    # Claim 001 - CLM-89921 (Auto Collision)
    # ==========================================================================
    
    # Needle questions
    TestCase(
        query="For claim CLM-89921, what was the exact accident time according to the police report?",
        ground_truth="October 14, 2023, at 11:30 PM",
        category="needle",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="For claim CLM-89921, what exact time did the hospital record for Sarah Connor's admission?",
        ground_truth="October 14, 2023, at 09:15 PM",
        category="needle",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="For claim CLM-89921, what was the specialized hazmat towing fee amount listed in the repair estimate?",
        ground_truth="$452.50",
        category="needle",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="For claim CLM-89921, what exclusion code might affect this claim and why?",
        ground_truth="Exclusion Code EX-9982 relating to unauthorized commercial use of a personal vehicle, due to social media posts showing the insured's Camry used for freelance package pickups",
        category="needle",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="For claim CLM-89921, which witness (specific name) may have been the actual driver according to inconsistencies?",
        ground_truth="Sarah O'Connor-Smith",
        category="needle",
        claim_id="CLM-89921",
    ),
    
    # Summary questions
    TestCase(
        query="For claim CLM-89921, what was the general sequence of events in the multi-vehicle collision?",
        ground_truth="Sarah Connor's vehicle was traveling at 45 mph on Highway 101 when she observed sudden brake lights from a black SUV ahead. She braked hard but made contact with the SUV's rear bumper. The SUV swerved into an adjacent lane hitting a Honda Civic, while a delivery van collided into Connor's Camry from behind.",
        category="summary",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="For claim CLM-89921, how did witness statements contribute to confusion about who was driving the insured vehicle?",
        ground_truth="Sarah O'Connor-Smith claimed to be in the passenger seat but her phrasing suggested she may have been controlling the wheel temporarily. She also claimed to be the driver at the moment of impact despite registration records listing only Sarah Connor. This created uncertainty for liability determination.",
        category="summary",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="For claim CLM-89921, what types of damages and injuries resulted from the collision?",
        ground_truth="Vehicle damages: front-end impact to Camry (bumper, hood, grille), rear-end compression, airbags deployed, fuel leakage. Injuries: Sarah Connor had chest tightness, clavicle bruising from seatbelt, and minor concussion symptoms (dizziness, light sensitivity).",
        category="summary",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="For claim CLM-89921, what overall factors influenced potential liability across the four vehicles?",
        ground_truth="Factors include: SUV's malfunctioning brake lights (cited), delivery van driver's GPS distraction (warning issued), construction zone lane merging, insured vehicle following too closely for nighttime conditions, and uncertainty about who was actually driving the insured vehicle.",
        category="summary",
        claim_id="CLM-89921",
    ),
    
    # ==========================================================================
    # Claim 002 - CLM-44217-PD (Water Damage)
    # ==========================================================================
    
    # Needle questions
    TestCase(
        query="For claim CLM-44217-PD, what specific type of pipe ruptured behind the laundry room wall?",
        ground_truth="PEX line (leading to the washing machine intake valve)",
        category="needle",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="For claim CLM-44217-PD, how many square feet of flooring were affected?",
        ground_truth="380 square feet",
        category="needle",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="For claim CLM-44217-PD, what time did the smart home logs first show abnormal humidity increase?",
        ground_truth="2:15 AM",
        category="needle",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="For claim CLM-44217-PD, what is the combined estimated total for mitigation and restoration?",
        ground_truth="$6,692.00 before deductible (mitigation $2,302.00 + restoration $4,390.00)",
        category="needle",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="For claim CLM-44217-PD, which mitigation company arrived on the day of the incident?",
        ground_truth="EverDry Response Unit",
        category="needle",
        claim_id="CLM-44217-PD",
    ),
    
    # Summary questions
    TestCase(
        query="For claim CLM-44217-PD, what caused the water intrusion and what major areas of the home were affected?",
        ground_truth="A PEX line leading to the washing machine intake valve ruptured longitudinally. Major areas affected: 380 sq ft of lower level flooring (laminate saturated), laundry room and guest bedroom shared wall, subfloor spaces, crawl area, and wall insulation.",
        category="summary",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="For claim CLM-44217-PD, how did the mitigation team address the structural and moisture problems?",
        ground_truth="EverDry Response Unit performed water extraction, installed dehumidifiers and HEPA air scrubbers, created containment barriers, demolished affected wall sections for airflow, removed saturated insulation, and applied initial mold treatment.",
        category="summary",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="For claim CLM-44217-PD, why did the adjuster approve Additional Living Expenses (ALE)?",
        ground_truth="Due to floor removal, dehumidification equipment noise levels, and mold concerns, the home's lower level became partially uninhabitable for a family with children. The insured relocated to a nearby short-term rental, and ALE costs were documented and deemed reasonable.",
        category="summary",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="For claim CLM-44217-PD, how did humidity data influence the adjuster's interpretation of the timeline?",
        ground_truth="Digital thermostat logs showed steady humidity rise starting around 2:15 AM, four hours before the insured claims discovery at 6:40 AM. This suggested minor seepage that escalated rapidly once the pipe failed fully, though still likely within coverage guidelines.",
        category="summary",
        claim_id="CLM-44217-PD",
    ),
    
    # ==========================================================================
    # Claim 003 - CLM-77182-CM (Theft)
    # ==========================================================================
    
    # Needle questions
    TestCase(
        query="For claim CLM-77182-CM, through which entrance did the intruder force entry?",
        ground_truth="Rear loading entrance door (forced open with pry-bar, strike plate showed evidence of pry-bar usage)",
        category="needle",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="For claim CLM-77182-CM, what was the total estimated replacement cost of equipment before depreciation?",
        ground_truth="$22,515.00",
        category="needle",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="For claim CLM-77182-CM, what type of vehicle was captured idling near the building during the theft?",
        ground_truth="Mid-2000s silver Nissan Frontier",
        category="needle",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="For claim CLM-77182-CM, how much was the policy deductible for this commercial claim?",
        ground_truth="$5,000 per occurrence",
        category="needle",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="What equipment was stolen in claim CLM-77182-CM?",
        ground_truth="RED Komodo 6K Camera, Canon RF 24-70mm lens, DJI Ronin RS3 Pro Gimbal, MacBook Pro 16\" M2 Max, Sennheiser EW-D Wireless Kit, Samsung 2TB SSD",
        category="needle",
        claim_id="CLM-77182-CM",
    ),
    
    # Summary questions
    TestCase(
        query="For claim CLM-77182-CM, what was the overall nature of the incident and how did the intruder access the premises?",
        ground_truth="A break-in at Lighthouse Creative Media's studio where high-value production equipment was stolen. The intruder accessed through the rear loading entrance by forcing the door with a pry-bar, causing moderate splintering to the door frame.",
        category="summary",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="For claim CLM-77182-CM, how did the theft disrupt business operations for Lighthouse Creative Media?",
        ground_truth="Three projects were impacted: commercial shoot for Pacific Coast Coffee Roasters delayed 5 days, wedding videography required emergency rental equipment, and agency project editing was delayed due to missing SSD with working files requiring backup reconstruction. Loss of billable hours estimated at $3,400 plus $1,180 emergency rental costs.",
        category="summary",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="For claim CLM-77182-CM, what was the adjuster's overall conclusion regarding fraud or inconsistencies?",
        ground_truth="No inconsistencies between claimed losses and company's equipment inventory. Previous invoices and photos of equipment in use matched the missing items. No red flags of intentional misrepresentation detected. Claim approved under Equipment Floater and Business Property sections.",
        category="summary",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="For claim CLM-77182-CM, why were some of the security cameras unable to capture the full incident?",
        ground_truth="Interior security cameras were not operational due to an ongoing system upgrade at the time of the break-in. The external camera only partially captured suspicious activity and the perpetrator's approach was not visible until exit.",
        category="summary",
        claim_id="CLM-77182-CM",
    ),
    
    # ==========================================================================
    # MCP Tool Questions - Policy Limit Validation
    # These questions REQUIRE the validate_policy_limit tool for computation
    # ==========================================================================
    TestCase(
        query="Use the policy validation tool to check: Is $9,766.90 within a $100,000 policy limit?",
        ground_truth="Yes, the claim is within policy limits. $9,766.90 is 9.77% of the $100,000 limit, with $90,233.10 remaining coverage. Risk level: Low.",
        category="mcp",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="Validate using the policy tool: Does a claimed amount of $12,735 exceed an equipment floater limit of $95,000?",
        ground_truth="No, the claim is within policy limits. $12,735 is 13.4% of the $95,000 limit, with $82,265 remaining coverage. Risk level: Low.",
        category="mcp",
        claim_id="CLM-77182-CM",
    ),
    TestCase(
        query="Use validate_policy_limit to check if $6,692 is within a dwelling coverage limit of $450,000.",
        ground_truth="Yes, the claim is within policy limits. $6,692 is 1.49% of the $450,000 limit, with $443,308 remaining coverage. Risk level: Low.",
        category="mcp",
        claim_id="CLM-44217-PD",
    ),
    TestCase(
        query="Run the policy validation tool: Is a claim of $150,000 within a $100,000 property damage limit?",
        ground_truth="No, the claim EXCEEDS policy limits. $150,000 is $50,000 over the $100,000 limit. Risk level: HIGH. Action required: Review for partial coverage or denial.",
        category="mcp",
        claim_id="CLM-89921",
    ),
    TestCase(
        query="Using the MCP policy tool, what is the risk level if claiming $80,000 against a $100,000 limit?",
        ground_truth="Claim is within policy limits. $80,000 is 80% of the $100,000 limit, with $20,000 remaining coverage. Risk level: High (due to high utilization).",
        category="mcp",
        claim_id="CLM-89921",
    ),
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def create_evaluators():
    """Create the three evaluators with the judge LLM."""
    judge_llm = OpenAI(model=JUDGE_MODEL, temperature=0)
    
    correctness = CorrectnessEvaluator(llm=judge_llm)
    relevancy = RelevancyEvaluator(llm=judge_llm)
    faithfulness = FaithfulnessEvaluator(llm=judge_llm)
    
    return correctness, relevancy, faithfulness


def evaluate_response(
    query: str,
    response: str,
    ground_truth: str,
    contexts: list[str],
    evaluators: tuple,
) -> dict:
    """Evaluate a single response using all three metrics."""
    correctness, relevancy, faithfulness = evaluators
    
    results = {}
    
    # Correctness: Does answer match ground truth?
    try:
        correctness_result = correctness.evaluate(
            query=query,
            response=response,
            reference=ground_truth,
        )
        results["correctness"] = {
            "score": correctness_result.score,
            "passing": correctness_result.passing,
            "feedback": correctness_result.feedback,
        }
    except Exception as e:
        results["correctness"] = {"score": 0, "passing": False, "feedback": str(e)}
    
    # Relevancy: Is the retrieved context relevant?
    try:
        relevancy_result = relevancy.evaluate(
            query=query,
            response=response,
            contexts=contexts,
        )
        results["relevancy"] = {
            "score": relevancy_result.score if relevancy_result.score else 0,
            "passing": relevancy_result.passing,
            "feedback": relevancy_result.feedback,
        }
    except Exception as e:
        results["relevancy"] = {"score": 0, "passing": False, "feedback": str(e)}
    
    # Faithfulness: Is the answer grounded in context?
    try:
        faithfulness_result = faithfulness.evaluate(
            query=query,
            response=response,
            contexts=contexts,
        )
        results["faithfulness"] = {
            "score": faithfulness_result.score if faithfulness_result.score else 0,
            "passing": faithfulness_result.passing,
            "feedback": faithfulness_result.feedback,
        }
    except Exception as e:
        results["faithfulness"] = {"score": 0, "passing": False, "feedback": str(e)}
    
    return results


async def run_evaluation(agent, test_cases: list[TestCase]) -> dict:
    """Run full evaluation on all test cases."""
    evaluators = create_evaluators()
    
    all_results = []
    category_scores = {"needle": [], "summary": [], "mcp": []}
    
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing: {test_case.query[:60]}...")
        
        try:
            # Track tool calls for this query
            tools_used = []
            
            # Get agent response using async run method with event streaming
            handler = agent.run(user_msg=test_case.query)
            
            # Capture tool calls from events
            async for event in handler.stream_events():
                if hasattr(event, 'tool_name'):
                    tool_info = {
                        "tool": event.tool_name,
                        "args": getattr(event, 'tool_kwargs', {})
                    }
                    if tool_info not in tools_used:
                        tools_used.append(tool_info)
                    print(f"   🔧 Tool: {event.tool_name}")
            
            # Get final response
            response = await handler
            response_text = str(response)
            
            # Extract contexts from the response (if available)
            contexts = []
            if hasattr(response, "source_nodes"):
                contexts = [node.text for node in response.source_nodes]
            elif hasattr(response, "sources"):
                contexts = [str(s) for s in response.sources]
            
            # If no contexts available, use the response itself
            if not contexts:
                contexts = [response_text]
            
            # Evaluate
            eval_results = evaluate_response(
                query=test_case.query,
                response=response_text,
                ground_truth=test_case.ground_truth,
                contexts=contexts,
                evaluators=evaluators,
            )
            
            result = {
                "query": test_case.query,
                "claim_id": test_case.claim_id,
                "category": test_case.category,
                "ground_truth": test_case.ground_truth,
                "response": response_text,
                "tools_used": tools_used,  # NEW: Track which tools were called
                "evaluation": eval_results,
            }
            
            all_results.append(result)
            
            # Track scores by category
            if eval_results["correctness"].get("score"):
                category_scores[test_case.category].append(
                    eval_results["correctness"]["score"]
                )
            
            # Print summary
            c_score = eval_results["correctness"].get("score", "N/A")
            r_score = eval_results["relevancy"].get("score", "N/A")
            f_score = eval_results["faithfulness"].get("score", "N/A")
            tools_list = " -> ".join([t.get("tool", "") for t in tools_used]) if tools_used else "none"
            print(f"   Tools: {tools_list}")
            print(f"   Correctness: {c_score} | Relevancy: {r_score} | Faithfulness: {f_score}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            all_results.append({
                "query": test_case.query,
                "claim_id": test_case.claim_id,
                "category": test_case.category,
                "error": str(e),
            })
    
    # Calculate aggregate statistics
    aggregate = {
        "total_tests": len(test_cases),
        "successful_tests": len([r for r in all_results if "error" not in r]),
        "category_averages": {},
    }
    
    for category, scores in category_scores.items():
        if scores:
            aggregate["category_averages"][category] = {
                "average_correctness": sum(scores) / len(scores),
                "count": len(scores),
            }
    
    # Calculate overall metrics
    all_correctness = []
    all_relevancy = []
    all_faithfulness = []
    
    for result in all_results:
        if "evaluation" in result:
            if result["evaluation"]["correctness"].get("score"):
                all_correctness.append(result["evaluation"]["correctness"]["score"])
            if result["evaluation"]["relevancy"].get("score"):
                all_relevancy.append(result["evaluation"]["relevancy"]["score"])
            if result["evaluation"]["faithfulness"].get("score"):
                all_faithfulness.append(result["evaluation"]["faithfulness"]["score"])
    
    aggregate["overall_averages"] = {
        "correctness": sum(all_correctness) / len(all_correctness) if all_correctness else 0,
        "relevancy": sum(all_relevancy) / len(all_relevancy) if all_relevancy else 0,
        "faithfulness": sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0,
    }
    
    return {
        "results": all_results,
        "aggregate": aggregate,
    }


def print_summary(evaluation_data: dict):
    """Print a formatted summary of evaluation results."""
    aggregate = evaluation_data["aggregate"]
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Tests: {aggregate['total_tests']}")
    print(f"Successful: {aggregate['successful_tests']}")
    
    print("\n--- Overall Averages ---")
    overall = aggregate["overall_averages"]
    print(f"  Correctness:  {overall['correctness']:.2f}")
    print(f"  Relevancy:    {overall['relevancy']:.2f}")
    print(f"  Faithfulness: {overall['faithfulness']:.2f}")
    
    print("\n--- By Category ---")
    for category, stats in aggregate["category_averages"].items():
        print(f"  {category.upper()}: Avg Correctness = {stats['average_correctness']:.2f} ({stats['count']} tests)")
    
    print("\n" + "=" * 70)


def get_user_selection() -> list[TestCase]:
    """Get user input to select which questions to evaluate."""
    general_cases = [tc for tc in TEST_CASES if tc.category in ("needle", "summary")]
    mcp_cases = [tc for tc in TEST_CASES if tc.category == "mcp"]
    
    print("\n" + "=" * 50)
    print("Insurance Claim RAG Evaluation")
    print("=" * 50)
    print(f"\nAvailable test cases:")
    print(f"  1. General questions (needle + summary): {len(general_cases)} available")
    print(f"  2. MCP questions (policy validation): {len(mcp_cases)} available")
    print(f"  3. All questions: {len(TEST_CASES)} total")
    print()
    
    while True:
        choice = input("Select question type (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    if choice == "1":
        max_count = len(general_cases)
        while True:
            count_input = input(f"How many general questions? (1-{max_count}, or 'all'): ").strip().lower()
            if count_input == "all":
                return general_cases
            try:
                count = int(count_input)
                if 1 <= count <= max_count:
                    return general_cases[:count]
                print(f"Please enter a number between 1 and {max_count}.")
            except ValueError:
                print("Invalid input. Enter a number or 'all'.")
    
    elif choice == "2":
        max_count = len(mcp_cases)
        while True:
            count_input = input(f"How many MCP questions? (1-{max_count}, or 'all'): ").strip().lower()
            if count_input == "all":
                return mcp_cases
            try:
                count = int(count_input)
                if 1 <= count <= max_count:
                    return mcp_cases[:count]
                print(f"Please enter a number between 1 and {max_count}.")
            except ValueError:
                print("Invalid input. Enter a number or 'all'.")
    
    else:
        return TEST_CASES


async def main():
    """Main evaluation entry point."""
    # Configure settings
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    # Get user selection
    selected_cases = get_user_selection()
    print(f"\nRunning evaluation on {len(selected_cases)} question(s)...")
    
    # Initialize system
    print("Initializing system for evaluation...")
    agent, documents = initialize_system()
    
    # Run evaluation
    evaluation_data = await run_evaluation(agent, selected_cases)
    
    # Save JSON results
    with open(RESULTS_PATH, "w") as f:
        json.dump(evaluation_data, f, indent=2, default=str)
    print(f"\nJSON results saved to: {RESULTS_PATH}")
    
    # Save CSV results
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Agent Answer", "Ground Truth", "Claim ID", "Category", "Tools Used", "Correctness Score"])
        
        for result in evaluation_data.get("results", []):
            question = result.get("query", "")
            answer = result.get("response", result.get("error", "ERROR"))
            ground_truth = result.get("ground_truth", "")
            claim_id = result.get("claim_id", "")
            category = result.get("category", "")
            
            # Format tools used
            tools_used = result.get("tools_used", [])
            tools_str = " -> ".join([t.get("tool", "") for t in tools_used]) if tools_used else "none"
            
            correctness = result.get("evaluation", {}).get("correctness", {}).get("score", "N/A")
            
            writer.writerow([question, answer, ground_truth, claim_id, category, tools_str, correctness])
    
    print(f"CSV results saved to: {CSV_PATH}")
    
    # Print summary
    print_summary(evaluation_data)


if __name__ == "__main__":
    asyncio.run(main())

