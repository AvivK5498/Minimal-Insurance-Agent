# 🚀 Quick Setup Instructions

## Prerequisites
- Python 3.10 or higher
- OpenAI API key
- `uv` package manager ([Install uv](https://github.com/astral-sh/uv))

## Installation (5 minutes)

### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup
```bash
cd midterm_assignment
```

### 3. Set your OpenAI API key
Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

Or export it:
```bash
export OPENAI_API_KEY=your-key-here
```

### 4. Install dependencies (automatic)
Dependencies are automatically installed by `uv` when you run the scripts.

---

## 🎯 Running the System

### Interactive Chat with Agent
```bash
uv run main.py
```

**Example queries:**
```
You: What is the claim ID for Sarah Connor's case?
You: For claim CLM-89921, what was the exact accident time?
You: Summarize the red flags in claim CLM-89921
```

Type `quit` or `exit` to stop.

---

### Run Evaluation Suite
```bash
uv run evaluation.py
```

This will:
1. Initialize the system (creates ChromaDB index)
2. Prompt you to select test cases (general, MCP, or all)
3. Run LLM-as-a-judge evaluation with 3 metrics
4. Save results to `evaluation_results.json` and `evaluation_results.csv`

Expected runtime: ~5-10 minutes for all 29 tests

---

### Run MCP Server (Standalone Testing)
```bash
uv run mcp_server.py
```

This starts the FastMCP server. Tools can be called directly or through the agent.

---

## 📂 File Structure

```
/midterm_assignment
├── main.py                    # Multi-agent system
├── evaluation.py              # Evaluation script (29 tests)
├── mcp_server.py              # MCP tools (policy validation)
├── README.md                  # Complete documentation
├── SETUP.md                   # This file
├── midterm_summary.pdf        # 1-page executive summary
├── diagram.png                # Architecture diagram
├── diagram.mmd                # Diagram source (Mermaid)
├── pyproject.toml             # Dependencies
├── uv.lock                    # Dependency lock file
├── .env                       # Your API key (create this)
├── .gitignore                 # Git ignore rules
└── data/
    ├── claim001.md            # Auto collision claim
    ├── claim002.md            # Water damage claim
    ├── claim003.md            # Theft claim
    └── metadata_cache.json    # Extracted metadata cache
```

**Auto-generated on first run:**
- `chroma_db/` - Vector database storage
- `evaluation_results.json` - Evaluation output
- `evaluation_results.csv` - CSV format results

---

## 🔍 Key Features to Test

1. **Claim Resolution**: Ask about claims by policy holder name or incident type
2. **Needle Queries**: Request specific facts (dates, costs, codes)
3. **Summary Queries**: Ask for overviews, timelines, red flags
4. **MCP Tool**: Request policy limit validation (system automatically uses the tool)

---

## ⚡ Troubleshooting

**Issue**: `OPENAI_API_KEY not found`  
**Fix**: Set your API key in `.env` or export as environment variable

**Issue**: `uv command not found`  
**Fix**: Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Issue**: Slow first run  
**Reason**: ChromaDB indexing on first run. Subsequent runs use cached index.

**Issue**: "Collection already exists" error  
**Fix**: Delete `chroma_db/` folder and restart

---

## 📊 Expected Evaluation Results

- **29 total tests** (16 needle, 8 summary, 5 MCP)
- **Correctness**: ~4.78/5.0
- **Relevancy**: ~0.98/1.0
- **Faithfulness**: ~0.99/1.0

---

## 📖 Documentation

For detailed architecture, design decisions, and evaluation methodology, see **README.md**.

For a quick overview, see **midterm_summary.pdf**.

