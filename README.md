# Minimal-RAG-Insurance-Agent

> **Educational Material**: This project was developed as part of an AI/ML course curriculum. It demonstrates practical implementation of multi-agent RAG systems for insurance claim analysis.

A multi-agent RAG (Retrieval-Augmented Generation) system for analyzing and querying insurance claims using LlamaIndex, hierarchical indexing, and MCP tool integration.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Segmentation & Chunking Strategy](#data-segmentation--chunking-strategy)
3. [Index Schemas](#index-schemas)
4. [Agent Design](#agent-design)
5. [MCP Integration](#mcp-integration)
6. [Evaluation Methodology](#evaluation-methodology)
7. [Setup & Execution](#setup--execution)
8. [Limitations & Trade-offs](#limitations--trade-offs)

---

## Architecture Overview

```
                         ┌─────────────────────┐
                         │     User Query      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Manager Agent     │
                         │   (gpt-4o Router)   │
                         │                     │
                         │  • Claim ID check   │
                         │  • Query routing    │
                         └──────────┬──────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌──────────▼──────────┐
│   Summary Expert    │  │   Needle Expert     │  │   MCP Policy Tool   │
│   (gpt-4o-mini)     │  │   (gpt-4o-mini)     │  │                     │
│                     │  │                     │  │  validate_policy_   │
│  High-level Q&A     │  │  Precise facts      │  │  limit()            │
│  Timeline overview  │  │  Dates, costs       │  │                     │
└──────────┬──────────┘  └──────────┬──────────┘  │  calculate_date_    │
           │                        │             │  difference()       │
┌──────────▼──────────┐  ┌──────────▼──────────┐  └─────────────────────┘
│   SummaryIndex      │  │  VectorStoreIndex   │
│   (MapReduce)       │  │  + AutoMerging      │
│                     │  │    Retriever        │
│  tree_summarize     │  │                     │
└─────────────────────┘  │  ChromaDB Backend   │
                         └─────────────────────┘
```

### Component Flow

1. **User Query** → Manager Agent receives the query
2. **Claim ID Validation** → Manager ensures a claim ID is provided
3. **Query Routing** → Manager selects the appropriate tool:
   - Summary Expert for high-level questions
   - Needle Expert for specific facts
   - MCP Tool for policy limit validation
4. **Response Synthesis** → Expert tool processes query and returns answer

---

## Data Segmentation & Chunking Strategy

### Hierarchical Structure

```
Claim Document
├── Level 1 (Root): 2048 tokens
│   └── Full sections with complete context
├── Level 2 (Intermediate): 512 tokens
│   └── Subsections for balanced reasoning
└── Level 3 (Leaf): 128 tokens
    └── Fine-grained facts for precision
```

### Chunk Size Rationale

| Level | Size | Purpose | Use Case |
|-------|------|---------|----------|
| **Root** | 2048 tokens | Broad context reconstruction | Understanding full incident narratives |
| **Intermediate** | 512 tokens | Balanced reasoning | Connecting related facts |
| **Leaf** | 128 tokens | Precision retrieval | Extracting specific dates, costs, codes |

### Why These Sizes?

1. **2048 tokens (Root)**: Matches typical section lengths in claim documents (Policy Details, Incident Report, etc.). Ensures complete context for summarization.

2. **512 tokens (Intermediate)**: Optimal for semantic similarity search. Balances specificity with enough context for the LLM to reason.

3. **128 tokens (Leaf)**: Captures atomic facts (e.g., "Accident Time: October 14, 2023, at 11:30 PM"). Essential for "needle-in-a-haystack" queries.

### Overlap Strategy

The `HierarchicalNodeParser` maintains parent-child relationships without explicit overlap. Instead, the `AutoMergingRetriever` dynamically merges leaf nodes into parent nodes when sufficient siblings are retrieved (40% threshold), providing contextual overlap on-demand.

### Recall Improvement

- **Multi-level retrieval**: Different chunk sizes capture different granularities
- **Auto-merging**: Automatically expands context when needed
- **Metadata filtering**: `claim_id` ensures retrieval from the correct document

---

## Index Schemas

### 1. Hierarchical Vector Index (Needle Index)

**Backend**: ChromaDB (persistent storage)

```python
# Storage structure
./chroma_db/
└── insurance_claims/  # Collection name
    ├── embeddings
    ├── metadata
    └── documents
```

**Metadata per chunk**:
- `claim_id`: Primary filter key
- `policy_holder`: Insured party name
- `source_file`: Original document filename

**Retriever**: `AutoMergingRetriever`
- `similarity_top_k=6`: Retrieve top 6 leaf nodes
- `simple_ratio_thresh=0.4`: Merge to parent if 40% of siblings retrieved

### 2. Summary Index (MapReduce)

**Response Mode**: `tree_summarize`

This implements the MapReduce pattern:
1. **Map**: Each retrieved chunk is summarized independently
2. **Reduce**: Summaries are combined into a final coherent response

Ideal for:
- Timeline overviews
- Claim summaries
- Red flag identification

---

## Agent Design

### Manager Agent (Router)

**Model**: `gpt-4o` (for sophisticated routing decisions)

**System Prompt**:
```
You are an Insurance Claim Manager Agent. Your role is to route user queries 
to specialized expert tools and provide accurate answers about insurance claims.

CRITICAL RULES:
1. You MUST obtain the claim_id from the user before retrieving any claim 
   information. If no claim_id is provided, ask the user to specify.
2. NEVER answer from memory - always use the appropriate tools.
3. For specific facts, dates, costs, or codes - use the needle_expert tool.
4. For summaries, timelines, or high-level overviews - use the summary_expert tool.
5. For validating if costs are within policy limits - use the validate_policy_limit tool.
```

### Summary Expert

**Model**: `gpt-4o-mini`  
**Tool Name**: `summary_expert`  
**Index**: SummaryIndex with tree_summarize

**Description**: "Useful for high-level questions, summaries of whole claims, timeline overviews, or understanding the overall situation."

### Needle Expert

**Model**: `gpt-4o-mini`  
**Tool Name**: `needle_expert`  
**Index**: VectorStoreIndex with AutoMergingRetriever

**Description**: "Useful for retrieving specific facts, dates, costs, codes, or fine-grained details from insurance claims."

---

## MCP Integration

### Overview

MCP (Model Context Protocol) extends the agent's capabilities beyond retrieval. The system includes an MCP server (`mcp_server.py`) with tools for:

1. **Policy Limit Validation**: Checks if claimed amounts exceed coverage limits
2. **Date Difference Calculation**: Computes time between claim events

### MCP Server (`mcp_server.py`)

```python
@mcp.tool()
def validate_policy_limit(claimed_amount: float, policy_limit: float) -> str:
    """
    Validates if a claimed amount is within policy limits.
    Returns structured analysis with risk assessment.
    """
```

### Usage Example

**Query**: "For claim CLM-89921, is the repair cost of $9,766.90 within the property damage limit of $100,000?"

**Tool Call**:
```python
validate_policy_limit(claimed_amount=9766.90, policy_limit=100000)
```

**Response**:
```
VALIDATION RESULT: Claim is WITHIN policy limits.
- Claimed Amount: $9,766.90
- Policy Limit: $100,000.00
- Coverage Used: 9.8%
- Remaining Coverage: $90,233.10
- Risk Level: Low
```

### Integration Pattern

The MCP tool is wrapped as a `FunctionTool` and provided to the Manager Agent alongside the expert query engines. This allows the agent to:
1. Retrieve cost information using the needle expert
2. Validate costs against policy limits using the MCP tool
3. Provide a comprehensive answer

---

## Evaluation Methodology

This project implements a comprehensive evaluation framework based on [Anthropic's guide to demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

### Three-Tier Grading System

| Grader Type | File | Purpose | Speed | Cost |
|-------------|------|---------|-------|------|
| **Code-Based** | `tests/test_code_based.py` | Deterministic checks (tool calls, format, content) | Fast | Free |
| **Model-Based** | `tests/test_model_based.py` | LLM-as-judge evaluation (correctness, relevancy, faithfulness) | Slow | $$$ |
| **Human-Based** | `tests/test_human_based.py` | CLI framework for human evaluation | Manual | Time |

### Code-Based Graders

Fast, cheap, and reproducible checks:

- **ToolCallGrader**: Validates correct tools are called in correct sequence
- **FormatGrader**: Verifies response contains expected patterns (dates, currency, claim IDs)
- **ResponseContentGrader**: Checks key facts appear in response

### Model-Based Graders (LLM-as-Judge)

Using `gpt-4o` as the judge model:

| Evaluator | Metric | Scale | Purpose |
|-----------|--------|-------|---------|
| `CorrectnessGrader` | Answer Correctness | 0-5 | Does the answer match ground truth? |
| `RelevancyGrader` | Context Relevancy | 0-1 | Is the retrieved context relevant to the query? |
| `FaithfulnessGrader` | Answer Faithfulness | 0-1 | Is the answer grounded in the retrieved context? |
| `CustomRubricGrader` | User-defined | 0-5 | Domain-specific evaluation criteria |

### Human-Based Graders

Gold standard for calibration and subjective evaluation:
- Interactive CLI for scoring responses
- Predefined rubrics for consistency
- Session management and result persistence

### Test Dataset

7 test cases covering all three claims and query types:

**Needle Questions (Specific Facts)**:
1. Exact accident time from police report (CLM-89921)
2. Specific pipe type that ruptured (CLM-44217-PD)
3. Total equipment replacement cost (CLM-77182-CM)

**Summary Questions (Broad Overview)**:
4. Multi-vehicle collision sequence (CLM-89921)
5. Water intrusion cause and affected areas (CLM-44217-PD)

**MCP Tool Questions**:
6. Repair cost within policy limit validation (CLM-89921)
7. Claim exceeding policy limit detection (CLM-89921)

### Running Tests

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

# Run legacy evaluation script
uv run evaluation.py
```

### Best Practices

Following Anthropic's recommendations:
1. **Grade outcomes, not paths**: Check what the agent produced, not exactly how it got there
2. **Isolated judges**: Use separate evaluators for each dimension
3. **Start small**: 20-50 test cases drawn from real failures
4. **Calibrate regularly**: Use human evaluation to calibrate model judges

---

## Setup & Execution

### Prerequisites

- Python 3.10+
- OpenAI API key
- `uv` package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/AvivK5498/Minimal-RAG-Insurance-Agent.git
cd Minimal-RAG-Insurance-Agent

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running the System

```bash
# Start interactive chat with the agent
uv run main.py

# Run evaluation suite
uv run evaluation.py

# Run MCP server standalone (for testing)
uv run mcp_server.py
```

### Example Queries

```
You: What is the claim ID for Sarah Connor's case?
Agent: Sarah Connor's case has claim ID CLM-89921.

You: For claim CLM-89921, what was the exact accident time?
Agent: According to the police report, the accident occurred on October 14, 2023, at 11:30 PM.

You: Summarize the red flags in claim CLM-89921.
Agent: [Provides summary of timeline inconsistencies, driver identity ambiguity, and commercial use concerns]
```

---

## Limitations & Trade-offs

### Current Limitations

1. **Single Vector Store**: All claims share one ChromaDB collection. For production, consider separate collections per claim type.

2. **Metadata Extraction**: LLM-based extraction may occasionally misparse unusual document formats.

3. **MCP Integration**: Currently uses local function wrapper. Full MCP server integration via stdio requires separate process management.

4. **Context Window**: Very large claims may exceed context limits during summarization.

### Trade-offs Made

| Decision | Trade-off |
|----------|-----------|
| **3-level hierarchy** | More storage overhead, but better retrieval flexibility |
| **gpt-4o for routing** | Higher cost, but more accurate tool selection |
| **gpt-4o-mini for experts** | Lower accuracy ceiling, but significantly reduced costs |
| **ChromaDB local** | Not production-scale, but portable for assignment submission |
| **LLM metadata extraction** | Slower initial load, but handles format variations |

### Future Improvements

1. **Streaming responses** for better UX
2. **Claim-specific collections** for better isolation
3. **Hybrid search** (semantic + keyword) for better recall
4. **Caching layer** for repeated queries
5. **Async batch processing** for large document sets

---

## File Structure

```
/Minimal-RAG-Insurance-Agent
├── pyproject.toml        # uv dependencies
├── .env                  # OPENAI_API_KEY (create from .env.example)
├── data/
│   ├── claim001.md       # Auto collision claim
│   ├── claim002.md       # Water damage claim
│   ├── claim003.md       # Theft claim
│   └── metadata_cache.json  # Extracted metadata cache
├── chroma_db/            # Persistent vector store
├── mcp_server.py         # FastMCP server with policy tools
├── main.py               # Manager Agent + Sub-Agents
├── evaluation.py         # Legacy evaluation script
├── tests/
│   ├── __init__.py           # Package marker
│   ├── conftest.py           # Shared fixtures and test cases
│   ├── test_code_based.py    # Code-based grader tests
│   ├── test_model_based.py   # Model-based grader tests
│   ├── test_human_based.py   # Human evaluation CLI
│   └── README.md             # Test suite documentation
├── .github/
│   └── workflows/
│       └── agent-evals.yml   # CI/CD workflow for tests
└── README.md             # This file
```

---

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)



