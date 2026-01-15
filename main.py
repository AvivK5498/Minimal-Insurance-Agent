# Run with: uv run main.py
"""
Insurance Claim Timeline Retrieval System
Multi-agent RAG system with hierarchical indexing, MapReduce summarization,
and MCP tool integration.
"""

import os
import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    Document,
    get_response_synthesizer,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import AgentWorkflow
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import MarkdownReader
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

# Load environment variables
load_dotenv()

# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    CYAN = "\033[36m"      # User queries
    GREEN = "\033[32m"     # User queries (alternative)
    YELLOW = "\033[33m"    # Agent responses
    BLUE = "\033[34m"      # Agent responses (alternative)
    MAGENTA = "\033[35m"   # System messages

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
METADATA_CACHE_PATH = DATA_DIR / "metadata_cache.json"

# Chunk sizes for hierarchical indexing (in tokens)
CHUNK_SIZES = [2048, 512, 128]  # Root, Intermediate, Leaf

# LLM Configuration
MANAGER_MODEL = "gpt-4o"
EXPERT_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# =============================================================================
# Metadata Extraction
# =============================================================================

class ClaimMetadata(BaseModel):
    """Schema for extracted claim metadata."""
    claim_id: str
    policy_holder: str


def load_metadata_cache() -> dict:
    """Load cached metadata from disk."""
    if METADATA_CACHE_PATH.exists():
        with open(METADATA_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_metadata_cache(cache: dict) -> None:
    """Save metadata cache to disk."""
    with open(METADATA_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def extract_metadata_llm(document_text: str, llm: OpenAI) -> ClaimMetadata:
    """Extract metadata from document using LLM structured output."""
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=ClaimMetadata,
        llm=llm,
        prompt_template_str=(
            "Extract the claim_id and policy_holder from this insurance claim document.\n"
            "The claim_id is usually at the top of the document (e.g., CLM-89921).\n"
            "The policy_holder is the name of the insured person or business.\n\n"
            "Document:\n{document_text}\n\n"
            "Extract the metadata:"
        ),
    )
    return program(document_text=document_text[:4000])  # Limit context for extraction


def load_documents_with_metadata() -> list[Document]:
    """Load all claim documents and attach extracted metadata."""
    reader = MarkdownReader()
    extraction_llm = OpenAI(model=EXTRACTION_MODEL, temperature=0.2)
    
    cache = load_metadata_cache()
    documents = []
    cache_updated = False
    
    for md_file in DATA_DIR.glob("*.md"):
        file_key = md_file.name
        
        # Load ALL document sections from the markdown file
        docs = reader.load_data(file=md_file)
        if not docs:
            continue
        
        # Extract metadata from the first section (contains claim ID and policy holder)
        first_doc = docs[0]
        
        # Check cache for metadata
        if file_key in cache:
            metadata = ClaimMetadata(**cache[file_key])
        else:
            # Extract metadata using LLM (use combined text from first few sections)
            combined_text = "\n".join([d.text for d in docs[:3]])
            print(f"Extracting metadata from {file_key}...")
            metadata = extract_metadata_llm(combined_text, extraction_llm)
            cache[file_key] = metadata.model_dump()
            cache_updated = True
        
        # Attach metadata to ALL document sections from this file
        for i, doc in enumerate(docs):
            doc.metadata["claim_id"] = metadata.claim_id
            doc.metadata["policy_holder"] = metadata.policy_holder
            doc.metadata["source_file"] = file_key
            doc.metadata["section_index"] = i
            documents.append(doc)
        
        print(f"Loaded: {file_key} | {len(docs)} sections | Claim ID: {metadata.claim_id} | Holder: {metadata.policy_holder}")
    
    # Save updated cache
    if cache_updated:
        save_metadata_cache(cache)
    
    return documents


# =============================================================================
# Indexing
# =============================================================================

def create_hierarchical_index(
    documents: list[Document],
    storage_context: StorageContext,
) -> VectorStoreIndex:
    """Create hierarchical index with 3-level chunking."""
    # Create hierarchical node parser with metadata inclusion
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=CHUNK_SIZES,
        include_metadata=True,
        include_prev_next_rel=True,
    )
    
    # Parse documents into hierarchical nodes
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # CRITICAL: Propagate document metadata to all nodes
    # The parser may not copy all metadata, so we do it explicitly
    doc_metadata_map = {doc.doc_id: doc.metadata for doc in documents}
    
    for node in nodes:
        # Get the source document's metadata
        if hasattr(node, 'ref_doc_id') and node.ref_doc_id in doc_metadata_map:
            source_metadata = doc_metadata_map[node.ref_doc_id]
            # Copy claim_id and policy_holder to node metadata
            node.metadata["claim_id"] = source_metadata.get("claim_id", "")
            node.metadata["policy_holder"] = source_metadata.get("policy_holder", "")
    
    print(f"  Created {len(nodes)} hierarchical nodes")
    
    # Verify metadata propagation
    sample_node = nodes[0] if nodes else None
    if sample_node:
        print(f"  Sample node metadata: claim_id={sample_node.metadata.get('claim_id', 'MISSING')}")
    
    # Get leaf nodes for indexing
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"  Leaf nodes for indexing: {len(leaf_nodes)}")
    
    # Store all nodes in docstore (needed for auto-merging)
    storage_context.docstore.add_documents(nodes)
    
    # Create vector index from leaf nodes
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    
    return index


def create_summary_index(documents: list[Document]) -> SummaryIndex:
    """Create summary index for high-level questions."""
    return SummaryIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
    )


# =============================================================================
# Query Engines with Metadata Filtering
# =============================================================================

from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator


def create_needle_query_engine_with_filter(
    index: VectorStoreIndex,
    storage_context: StorageContext,
    claim_id: Optional[str] = None,
) -> RetrieverQueryEngine:
    """Create query engine with AutoMergingRetriever and optional claim_id filter."""
    
    # Build filters if claim_id provided
    filters = None
    if claim_id:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="claim_id",
                    value=claim_id,
                    operator=FilterOperator.EQ,
                ),
            ]
        )
    
    # Create base retriever with filters
    # top_k=6 is sufficient since metadata filtering narrows to single claim (~28 nodes)
    base_retriever = index.as_retriever(
        similarity_top_k=12,
        filters=filters,
    )
    
    # Create auto-merging retriever
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=0.3,  # Merge if 30% of siblings retrieved
    )
    
    # Create response synthesizer
    response_synthesizer = get_response_synthesizer(
        llm=OpenAI(model=EXPERT_MODEL),
        response_mode="compact",
    )
    
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )


def create_summary_query_engine_with_filter(
    index: SummaryIndex,
    claim_id: Optional[str] = None,
) -> RetrieverQueryEngine:
    """Create query engine for broad summary questions with optional claim_id filter."""
    # Note: SummaryIndex doesn't support metadata filtering the same way
    # For now, we rely on the query to specify the claim
    return index.as_query_engine(
        llm=OpenAI(model=EXPERT_MODEL),
        response_mode="tree_summarize",  # MapReduce-style summarization
    )


# =============================================================================
# MCP Tool Integration
# =============================================================================

def validate_policy_limit(claimed_amount: float, policy_limit: float) -> str:
    """
    Validates if a claimed amount is within the policy coverage limits.
    
    Args:
        claimed_amount: The total amount being claimed in dollars
        policy_limit: The maximum coverage limit from the policy in dollars
    
    Returns:
        A structured analysis string indicating if the claim is within limits
    """
    if claimed_amount < 0:
        return "ERROR: Claimed amount cannot be negative."
    
    if policy_limit <= 0:
        return "ERROR: Policy limit must be a positive value."
    
    percentage_used = (claimed_amount / policy_limit) * 100
    remaining = policy_limit - claimed_amount
    
    if claimed_amount <= policy_limit:
        risk_level = "Low" if percentage_used < 50 else "Medium" if percentage_used < 80 else "High"
        return (
            f"VALIDATION RESULT: Claim is WITHIN policy limits.\n"
            f"- Claimed Amount: ${claimed_amount:,.2f}\n"
            f"- Policy Limit: ${policy_limit:,.2f}\n"
            f"- Coverage Used: {percentage_used:.1f}%\n"
            f"- Remaining Coverage: ${remaining:,.2f}\n"
            f"- Risk Level: {risk_level}"
        )
    else:
        excess = claimed_amount - policy_limit
        return (
            f"VALIDATION RESULT: Claim EXCEEDS policy limits.\n"
            f"- Claimed Amount: ${claimed_amount:,.2f}\n"
            f"- Policy Limit: ${policy_limit:,.2f}\n"
            f"- Amount Over Limit: ${excess:,.2f}\n"
            f"- Risk Level: HIGH\n"
            f"- Action Required: Review for partial coverage or denial."
        )


# =============================================================================
# Claim Registry & Resolution
# =============================================================================

@dataclass
class ClaimInfo:
    """Metadata for a single claim."""
    claim_id: str
    policy_holder: str
    incident_type: str
    policy_type: str
    keywords: list[str]  # Additional searchable terms


# Global claim registry - populated during initialization
_claim_registry: list[ClaimInfo] = []

# Global references for tools (set during initialization)
_hierarchical_index = None
_storage_context = None
_summary_index = None


def build_claim_registry(documents: list[Document]) -> list[ClaimInfo]:
    """Build a searchable registry from loaded documents (deduplicated by claim_id)."""
    # Group documents by claim_id
    claim_docs: dict[str, list[Document]] = {}
    for doc in documents:
        claim_id = doc.metadata.get("claim_id", "")
        if claim_id:
            if claim_id not in claim_docs:
                claim_docs[claim_id] = []
            claim_docs[claim_id].append(doc)
    
    registry = []
    
    for claim_id, docs in claim_docs.items():
        # Get policy holder from first doc
        policy_holder = docs[0].metadata.get("policy_holder", "")
        
        # Combine all document text to detect incident type
        combined_text = " ".join([d.text.lower() for d in docs])
        
        # Extract incident type from combined content
        if "theft" in combined_text or "stolen" in combined_text:
            incident_type = "theft"
        elif "water" in combined_text or "flood" in combined_text or "pipe" in combined_text:
            incident_type = "water_damage"
        elif "collision" in combined_text or "accident" in combined_text or "vehicle" in combined_text:
            incident_type = "auto_collision"
        else:
            incident_type = "unknown"
        
        # Determine policy type
        if "auto" in combined_text or "vehicle" in combined_text:
            policy_type = "auto"
        elif "homeowner" in combined_text or "residential" in combined_text:
            policy_type = "homeowners"
        elif "commercial" in combined_text or "business" in combined_text:
            policy_type = "commercial"
        else:
            policy_type = "unknown"
        
        # Build keywords for fuzzy matching
        keywords = [
            claim_id.lower(),
            policy_holder.lower(),
            incident_type,
            policy_type,
        ]
        # Add name parts
        keywords.extend(policy_holder.lower().split())
        
        registry.append(ClaimInfo(
            claim_id=claim_id,
            policy_holder=policy_holder,
            incident_type=incident_type,
            policy_type=policy_type,
            keywords=keywords,
        ))
    
    return registry


def resolve_claim(
    claim_id: Optional[str] = None,
    policy_holder: Optional[str] = None,
    incident_type: Optional[str] = None,
) -> str:
    """
    Resolves user-provided identifiers to a specific claim.
    Use this FIRST before querying to ensure you have the correct claim_id.
    
    Args:
        claim_id: Direct claim ID if known (e.g., 'CLM-89921')
        policy_holder: Name of the insured (e.g., 'Sarah Connor', 'Lighthouse')
        incident_type: Type of incident (e.g., 'theft', 'water_damage', 'auto_collision', 'accident')
    
    Returns:
        If exactly ONE match: Returns the claim_id to use.
        If MULTIPLE matches: Returns list of matching claims - you must ask user to specify.
        If NO matches: Returns error message.
    """
    global _claim_registry
    
    if not _claim_registry:
        return "ERROR: Claim registry not initialized"
    
    matches = []
    
    for claim in _claim_registry:
        score = 0
        
        # Direct claim_id match (highest priority)
        if claim_id:
            if claim_id.upper() == claim.claim_id.upper():
                return f"RESOLVED: {claim.claim_id} ({claim.policy_holder} - {claim.incident_type})"
            if claim_id.upper() in claim.claim_id.upper():
                score += 10
        
        # Policy holder match
        if policy_holder:
            holder_lower = policy_holder.lower()
            if holder_lower in claim.policy_holder.lower():
                score += 5
            # Check individual name parts
            for part in holder_lower.split():
                if part in claim.policy_holder.lower():
                    score += 2
        
        # Incident type match
        if incident_type:
            type_lower = incident_type.lower()
            # Normalize common terms
            type_mapping = {
                "theft": ["theft", "stolen", "burglary", "break-in"],
                "water_damage": ["water", "flood", "pipe", "leak", "water_damage"],
                "auto_collision": ["auto", "collision", "accident", "vehicle", "car", "crash"],
            }
            for canonical, variants in type_mapping.items():
                if type_lower in variants or canonical in type_lower:
                    if claim.incident_type == canonical:
                        score += 5
                    break
        
        if score > 0:
            matches.append((score, claim))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[0], reverse=True)
    
    if not matches:
        available = ", ".join([f"{c.claim_id} ({c.policy_holder})" for c in _claim_registry])
        return f"NO MATCH FOUND. Available claims: {available}"
    
    if len(matches) == 1:
        claim = matches[0][1]
        return f"RESOLVED: {claim.claim_id} ({claim.policy_holder} - {claim.incident_type})"
    
    # Multiple matches - check if top score is significantly higher
    if matches[0][0] > matches[1][0] * 1.5:
        claim = matches[0][1]
        return f"RESOLVED: {claim.claim_id} ({claim.policy_holder} - {claim.incident_type})"
    
    # Ambiguous - return all matches for user clarification
    match_list = "\n".join([
        f"  - {m[1].claim_id}: {m[1].policy_holder} ({m[1].incident_type})"
        for m in matches
    ])
    return f"AMBIGUOUS: Multiple claims match. Please specify which one:\n{match_list}"


# =============================================================================
# Agent Setup
# =============================================================================

MANAGER_SYSTEM_PROMPT = """You are an Insurance Claim Manager Agent. Your role is to route user queries to specialized expert tools and provide accurate answers about insurance claims.

WORKFLOW (MUST FOLLOW):
1. FIRST use resolve_claim to identify the exact claim from user's query (unless query is about policy validation with explicit amounts)
2. If resolution returns "AMBIGUOUS", ask user to specify which claim
3. If resolution returns "RESOLVED", extract the claim_id and proceed
4. Use needle_expert for specific facts (dates, costs, codes)
5. Use summary_expert for overviews and summaries
6. For policy limit validation questions:
   a. FIRST extract the claimed_amount from the claim document using needle_expert
   b. THEN extract the policy_limit from the claim document using needle_expert (look for "Coverage Limits", "Policy Limit", or similar sections)
   c. ONLY THEN call validate_policy_limit with the EXACT values extracted from the document
   d. NEVER guess, estimate, or use values not explicitly found in the claim document

CRITICAL RULES:
- NEVER skip the resolve_claim step (except for direct policy validation with explicit numbers)
- NEVER guess or assume a claim_id
- If ambiguous, ALWAYS ask user to clarify
- NEVER answer from memory - always use tools
- When validating policy limits, you MUST extract BOTH values from the claim document first
- NEVER use validate_policy_limit with guessed or assumed values
- If you cannot find exact values in the document, report that the information is not available
- Always include ALL costs (mitigation, restoration, ALE, etc.) when calculating total claimed amounts

MCP TOOL - validate_policy_limit:
- Use for ANY policy limit validation
- Requires: claimed_amount (float), policy_limit (float)
- BOTH values MUST be extracted from the claim document using needle_expert FIRST
- Returns: Validation result with risk level and remaining coverage
- Example workflow:
  1. needle_expert(claim_id='CLM-44217-PD', query='what is the total claimed amount including all costs?')
  2. needle_expert(claim_id='CLM-44217-PD', query='what are the policy coverage limits?')
  3. validate_policy_limit(claimed_amount=<extracted_value>, policy_limit=<extracted_value>)

Available Claim Types:
- Auto collision claims
- Water damage / property claims  
- Theft / commercial claims

Always provide clear, accurate responses based on the tool outputs."""


def needle_expert(claim_id: str, query: str) -> str:
    """
    Retrieves specific facts, dates, costs, or codes from a specific insurance claim.
    IMPORTANT: Use resolve_claim FIRST to get the correct claim_id.
    
    Args:
        claim_id: The EXACT claim ID from resolve_claim (e.g., 'CLM-89921')
        query: The specific question to answer about the claim
    
    Returns:
        The answer based on retrieved claim details
    """
    global _hierarchical_index, _storage_context
    
    if not _hierarchical_index or not _storage_context:
        return "ERROR: System not initialized"
    
    # Validate claim_id exists
    valid_ids = [c.claim_id for c in _claim_registry]
    if claim_id not in valid_ids:
        return f"ERROR: Invalid claim_id '{claim_id}'. Use resolve_claim first. Valid IDs: {valid_ids}"
    
    # Create query engine with claim_id filter
    engine = create_needle_query_engine_with_filter(
        _hierarchical_index,
        _storage_context,
        claim_id=claim_id,
    )
    
    response = engine.query(query)
    return str(response)


def summary_expert(claim_id: str, query: str) -> str:
    """
    Provides high-level summaries, timeline overviews, or broad analysis of a specific insurance claim.
    IMPORTANT: Use resolve_claim FIRST to get the correct claim_id.
    
    Args:
        claim_id: The EXACT claim ID from resolve_claim (e.g., 'CLM-89921')
        query: The summary or overview question about the claim
    
    Returns:
        A summary or overview based on the claim documents
    """
    global _summary_index
    
    if not _summary_index:
        return "ERROR: System not initialized"
    
    # Validate claim_id exists
    valid_ids = [c.claim_id for c in _claim_registry]
    if claim_id not in valid_ids:
        return f"ERROR: Invalid claim_id '{claim_id}'. Use resolve_claim first. Valid IDs: {valid_ids}"
    
    # Create query engine (add claim context to query for better filtering)
    engine = create_summary_query_engine_with_filter(_summary_index, claim_id)
    
    # Prepend claim_id to query for context
    enhanced_query = f"For claim {claim_id}: {query}"
    response = engine.query(enhanced_query)
    return str(response)


def create_manager_agent(
    hierarchical_index: VectorStoreIndex,
    storage_context: StorageContext,
    summary_index: SummaryIndex,
    documents: list[Document],
) -> AgentWorkflow:
    """Create the Manager Agent with access to expert tools and MCP tools."""
    global _hierarchical_index, _storage_context, _summary_index, _claim_registry
    
    # Set global references for tool functions
    _hierarchical_index = hierarchical_index
    _storage_context = storage_context
    _summary_index = summary_index
    
    # Build claim registry for resolution
    _claim_registry = build_claim_registry(documents)
    print(f"  Built claim registry with {len(_claim_registry)} claims:")
    for claim in _claim_registry:
        print(f"    - {claim.claim_id}: {claim.policy_holder} ({claim.incident_type})")
    
    # Create resolve_claim tool (MUST be called first)
    resolve_tool = FunctionTool.from_defaults(
        fn=resolve_claim,
        name="resolve_claim",
        description=(
            "MUST BE CALLED FIRST to identify the correct claim from user input. "
            "Accepts optional: claim_id, policy_holder, incident_type. "
            "Returns RESOLVED with claim_id if unique match, or AMBIGUOUS with options if multiple matches. "
            "Examples: resolve_claim(policy_holder='Sarah') or resolve_claim(incident_type='theft')"
        ),
    )
    
    # Create function tools with claim_id parameter
    needle_tool = FunctionTool.from_defaults(
        fn=needle_expert,
        name="needle_expert",
        description=(
            "Retrieves specific facts, dates, costs, codes from a claim. "
            "REQUIRES exact claim_id from resolve_claim. "
            "Example: needle_expert(claim_id='CLM-89921', query='what was the accident time?')"
        ),
    )
    
    summary_tool = FunctionTool.from_defaults(
        fn=summary_expert,
        name="summary_expert",
        description=(
            "Provides high-level summaries and timeline overviews of a claim. "
            "REQUIRES exact claim_id from resolve_claim. "
            "Example: summary_expert(claim_id='CLM-89921', query='summarize the red flags')"
        ),
    )
    
    # Create MCP tool (local function version for simplicity)
    policy_tool = FunctionTool.from_defaults(
        fn=validate_policy_limit,
        name="validate_policy_limit",
        description=(
            "Validates if a claimed amount is within policy coverage limits. "
            "CRITICAL: You MUST extract BOTH claimed_amount and policy_limit from the claim document "
            "using needle_expert BEFORE calling this function. NEVER guess or assume values. "
            "Use this when asked to check if costs exceed policy limits. "
            "Requires claimed_amount (float) and policy_limit (float) in dollars. "
            "Both values must be explicitly found in the claim document."
        ),
    )
    
    # Create manager agent using AgentWorkflow
    manager_llm = OpenAI(model=MANAGER_MODEL, temperature=0.2)
    
    # resolve_claim FIRST in the tools list to encourage using it first
    tools = [resolve_tool, needle_tool, summary_tool, policy_tool]
    
    agent = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=tools,
        llm=manager_llm,
        system_prompt=MANAGER_SYSTEM_PROMPT,
        verbose=True,
    )
    
    return agent


# =============================================================================
# Main Entry Point
# =============================================================================

def initialize_system():
    """Initialize the complete RAG system."""
    print("=" * 60)
    print("Insurance Claim Timeline Retrieval System")
    print("=" * 60)
    
    # Configure global settings
    Settings.llm = OpenAI(model=EXPERT_MODEL)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    
    # Load documents with metadata
    print("\n[1/5] Loading documents with metadata extraction...")
    documents = load_documents_with_metadata()
    print(f"Loaded {len(documents)} documents")
    
    # Setup ChromaDB
    print("\n[2/5] Setting up ChromaDB vector store...")
    CHROMA_DIR.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Delete existing collection to ensure fresh index
    # (Required because docstore is in-memory and loses parent-child relationships on restart)
    try:
        chroma_client.delete_collection(name="insurance_claims")
        print("  Cleared existing index for fresh rebuild...")
    except Exception:
        pass  # Collection doesn't exist yet
    
    # Create fresh collection
    chroma_collection = chroma_client.create_collection(
        name="insurance_claims",
        metadata={"hnsw:space": "cosine"},
    )
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create hierarchical index
    print("\n[3/5] Creating hierarchical index (3-level chunking)...")
    hierarchical_index = create_hierarchical_index(documents, storage_context)
    
    # Create summary index
    print("\n[4/5] Creating summary index (MapReduce)...")
    summary_index = create_summary_index(documents)
    
    # Create manager agent with access to indexes and claim registry
    print("\n[5/5] Setting up agent with smart claim resolution...")
    agent = create_manager_agent(hierarchical_index, storage_context, summary_index, documents)
    
    print("\n" + "=" * 60)
    print("System initialized successfully!")
    print("=" * 60)
    
    return agent, documents


async def chat_loop(agent: AgentWorkflow):
    """Interactive chat loop with the agent."""
    print("\nYou can now ask questions about insurance claims.")
    print("Type 'quit' or 'exit' to stop.")
    print("(Verbose mode: showing agent reasoning and tool calls)\n")
    
    while True:
        try:
            # Display user prompt in cyan
            query = input(f"{Colors.CYAN}You: {Colors.RESET}").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            
            # Display user query in cyan for visibility
            print(f"{Colors.CYAN}{query}{Colors.RESET}")
            print(f"\n{Colors.MAGENTA}--- Agent Processing ---{Colors.RESET}")
            
            # Stream events to see what the agent is doing
            handler = agent.run(user_msg=query)
            
            async for event in handler.stream_events():
                # Print different event types
                event_type = type(event).__name__
                
                if hasattr(event, 'tool_name'):
                    print(f"  🔧 Tool Call: {event.tool_name}")
                    if hasattr(event, 'tool_kwargs'):
                        print(f"     Args: {event.tool_kwargs}")
                
                if hasattr(event, 'tool_output'):
                    output = str(event.tool_output)[:200]
                    print(f"  📤 Tool Output: {output}...")
                
                if hasattr(event, 'response') and event_type == 'AgentOutput':
                    pass  # We'll get the final response below
                
                if hasattr(event, 'msg') and 'thought' in event_type.lower():
                    print(f"  💭 Thought: {event.msg}")
            
            # Get final response
            response = await handler
            print(f"\n{Colors.MAGENTA}--- Final Response ---{Colors.RESET}")
            # Display agent response in yellow
            print(f"{Colors.YELLOW}Agent: {response}{Colors.RESET}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            import traceback
            print(f"\nError: {e}")
            traceback.print_exc()
            print()


async def main():
    """Main entry point."""
    agent, documents = initialize_system()
    
    # Start chat loop
    await chat_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())

