# Market Analysis Wizard

Multi-agent market intelligence tool — LangGraph pipeline with real-time SSE streaming. Enter a business idea, watch 6 AI agents research it live, get a full analyst report with TAM/SAM/SOM, competitor landscape, SWOT, and go-to-market strategy.

## Architecture

```
FastAPI backend
└── LangGraph 6-node pipeline
    ├── Query Planner     → breaks idea into 6 targeted research angles
    ├── Web Researcher    → Tavily search across all angles in parallel
    ├── Competitor Intel  → extracts competitor landscape from findings
    ├── Market Sizing     → TAM / SAM / SOM with CAGR estimation
    ├── SWOT Synthesiser  → grounded analysis from research findings
    └── Report Writer     → structured intelligence report with citations
```

SSE streaming delivers live agent events to the frontend as each node completes.

## Stack

- **Backend:** FastAPI + LangGraph + LangChain-Anthropic + Tavily
- **LLM:** Claude Sonnet (Anthropic)
- **Search:** Tavily (free tier available)
- **Frontend:** Vanilla HTML/CSS/JS — single file, no build step

## Setup

```bash
git clone https://github.com/aman-24052001/market-analysis-wizard
cd market-analysis-wizard
pip install -r requirements.txt
cp .env.example .env   # add your API keys
uvicorn main:app --reload
```

Open `http://localhost:8000`

## API Keys

- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com)
- `TAVILY_API_KEY` — [app.tavily.com](https://app.tavily.com) (free tier: 1000 searches/month)

## Author

Aman Kumar — [github.com/aman-24052001](https://github.com/aman-24052001)
