"""
Market Analysis Pipeline — LangGraph 6-node multi-agent pipeline
Nodes: QueryPlanner → WebResearcher → CompetitorIntel → MarketSizing → SWOTSynthesiser → ReportWriter
"""
from __future__ import annotations
import asyncio
import json
import os
from typing import Any, AsyncIterator
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from tavily import AsyncTavilyClient


# ── State ──────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    idea: str
    queries: list[str]
    web_findings: list[dict]
    competitor_findings: list[dict]
    market_sizing: dict
    swot: dict
    report: dict
    events: list[dict]   # SSE log — appended by each node


# ── Helpers ────────────────────────────────────────────────────────────────────

def _llm(temperature: float = 0.3) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=temperature,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=4096,
    )


def _evt(node: str, msg: str, data: Any = None) -> dict:
    return {"node": node, "msg": msg, "data": data}


async def _tavily_search(client: AsyncTavilyClient, query: str, max_results: int = 5) -> list[dict]:
    try:
        resp = await client.search(query, max_results=max_results, search_depth="advanced", include_raw_content=False)
        return resp.get("results", [])
    except Exception as e:
        return [{"title": "Search error", "url": "", "content": str(e)}]


# ── Node 1: Query Planner ──────────────────────────────────────────────────────

async def query_planner(state: PipelineState) -> PipelineState:
    idea = state["idea"]
    events = list(state.get("events", []))
    events.append(_evt("query_planner", f"Planning research strategy for: {idea}"))

    llm = _llm()
    resp = await llm.ainvoke([
        SystemMessage(content="You are a market research strategist. Output ONLY valid JSON, no markdown."),
        HumanMessage(content=f"""
For the business idea: "{idea}"

Generate exactly 6 targeted search queries that together will cover:
1. Overall market size and growth rate
2. Key competitors and their market share
3. Target customer demographics and pain points
4. Recent industry trends and news (2024-2026)
5. Regulatory environment and barriers to entry
6. Geographic market opportunities

Return JSON: {{"queries": ["query1", "query2", "query3", "query4", "query5", "query6"]}}
""")
    ])

    try:
        raw = resp.content.strip().replace("```json", "").replace("```", "")
        queries = json.loads(raw)["queries"]
    except Exception:
        queries = [
            f"{idea} market size 2025",
            f"{idea} competitors market share",
            f"{idea} target customers demographics",
            f"{idea} industry trends 2025 2026",
            f"{idea} regulations entry barriers",
            f"{idea} best cities regions opportunities",
        ]

    events.append(_evt("query_planner", f"Generated {len(queries)} research queries", queries))
    return {**state, "queries": queries, "events": events}


# ── Node 2: Web Researcher ─────────────────────────────────────────────────────

async def web_researcher(state: PipelineState) -> PipelineState:
    events = list(state.get("events", []))
    events.append(_evt("web_researcher", f"Running {len(state['queries'])} parallel web searches"))

    tavily = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    tasks = [_tavily_search(tavily, q) for q in state["queries"]]
    results_per_query = await asyncio.gather(*tasks)

    all_findings = []
    for query, results in zip(state["queries"], results_per_query):
        for r in results:
            all_findings.append({
                "query": query,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:600],
            })
        events.append(_evt("web_researcher", f"✓ '{query[:55]}...' → {len(results)} sources"))

    events.append(_evt("web_researcher", f"Total: {len(all_findings)} sources collected"))
    return {**state, "web_findings": all_findings, "events": events}


# ── Node 3: Competitor Intel ───────────────────────────────────────────────────

async def competitor_intel(state: PipelineState) -> PipelineState:
    events = list(state.get("events", []))
    events.append(_evt("competitor_intel", "Extracting and analysing competitor landscape"))

    findings_text = "\n\n".join(
        f"[{f['title']}]\n{f['content']}\nSource: {f['url']}"
        for f in state["web_findings"][:15]
    )

    llm = _llm()
    resp = await llm.ainvoke([
        SystemMessage(content="You are a competitive intelligence analyst. Output ONLY valid JSON, no markdown fences."),
        HumanMessage(content=f"""
Business idea: "{state['idea']}"

Web research findings:
{findings_text}

Extract a detailed competitor analysis. For each competitor provide realistic estimates based on the research.

Return JSON:
{{
  "competitors": [
    {{
      "name": "Company Name",
      "market_share_pct": 15,
      "positioning": "...",
      "strengths": ["..."],
      "weaknesses": ["..."],
      "recent_moves": "...",
      "source_url": "..."
    }}
  ],
  "competitive_intensity": "low|medium|high",
  "key_differentiators": ["..."]
}}

Include 4-6 real competitors. Be specific and grounded in the research.
""")
    ])

    try:
        raw = resp.content.strip().replace("```json", "").replace("```", "")
        competitor_data = json.loads(raw)
    except Exception:
        competitor_data = {"competitors": [], "competitive_intensity": "medium", "key_differentiators": []}

    events.append(_evt("competitor_intel",
        f"Identified {len(competitor_data.get('competitors', []))} competitors — intensity: {competitor_data.get('competitive_intensity', 'medium')}",
        [c["name"] for c in competitor_data.get("competitors", [])]))

    return {**state, "competitor_findings": competitor_data, "events": events}


# ── Node 4: Market Sizing ──────────────────────────────────────────────────────

async def market_sizing(state: PipelineState) -> PipelineState:
    events = list(state.get("events", []))
    events.append(_evt("market_sizing", "Estimating TAM / SAM / SOM and growth trajectory"))

    findings_text = "\n\n".join(
        f"[{f['title']}]\n{f['content']}"
        for f in state["web_findings"][:12]
    )

    llm = _llm()
    resp = await llm.ainvoke([
        SystemMessage(content="You are a market sizing analyst. Output ONLY valid JSON, no markdown."),
        HumanMessage(content=f"""
Business idea: "{state['idea']}"

Research findings:
{findings_text}

Provide a rigorous market sizing analysis with TAM/SAM/SOM framework.

Return JSON:
{{
  "tam": {{"value": "₹X Cr / $X B", "basis": "how you calculated this"}},
  "sam": {{"value": "₹X Cr / $X B", "basis": "..."}},
  "som": {{"value": "₹X Cr / $X B", "basis": "realistic 3-year capture"}},
  "cagr": "X%",
  "cagr_period": "2024-2029",
  "market_stage": "emerging|growing|mature|declining",
  "key_growth_drivers": ["...", "..."],
  "key_risks": ["...", "..."],
  "target_segments": [
    {{"segment": "...", "size_pct": 30, "description": "..."}}
  ]
}}
""")
    ])

    try:
        raw = resp.content.strip().replace("```json", "").replace("```", "")
        sizing = json.loads(raw)
    except Exception:
        sizing = {"tam": {"value": "N/A", "basis": ""}, "sam": {"value": "N/A", "basis": ""}, "som": {"value": "N/A", "basis": ""}}

    events.append(_evt("market_sizing",
        f"TAM: {sizing.get('tam', {}).get('value', '?')} | SAM: {sizing.get('sam', {}).get('value', '?')} | CAGR: {sizing.get('cagr', '?')}"))

    return {**state, "market_sizing": sizing, "events": events}


# ── Node 5: SWOT Synthesiser ───────────────────────────────────────────────────

async def swot_synthesiser(state: PipelineState) -> PipelineState:
    events = list(state.get("events", []))
    events.append(_evt("swot_synthesiser", "Synthesising SWOT from all research findings"))

    llm = _llm()
    resp = await llm.ainvoke([
        SystemMessage(content="You are a strategic analyst. Output ONLY valid JSON, no markdown."),
        HumanMessage(content=f"""
Business idea: "{state['idea']}"

Competitor landscape: {json.dumps(state['competitor_findings'], indent=2)[:1500]}
Market sizing: {json.dumps(state['market_sizing'], indent=2)[:800]}

Generate a deep, grounded SWOT analysis. Each point must be specific and actionable — no generic platitudes.

Return JSON:
{{
  "strengths": [
    {{"point": "...", "evidence": "why this is a genuine strength based on research"}}
  ],
  "weaknesses": [
    {{"point": "...", "evidence": "..."}}
  ],
  "opportunities": [
    {{"point": "...", "evidence": "..."}}
  ],
  "threats": [
    {{"point": "...", "evidence": "..."}}
  ],
  "strategic_priority": "The single most important thing to focus on in year 1"
}}

Provide 3-4 points per quadrant.
""")
    ])

    try:
        raw = resp.content.strip().replace("```json", "").replace("```", "")
        swot = json.loads(raw)
    except Exception:
        swot = {"strengths": [], "weaknesses": [], "opportunities": [], "threats": [], "strategic_priority": ""}

    events.append(_evt("swot_synthesiser",
        f"SWOT complete — {len(swot.get('strengths',[]))}S {len(swot.get('weaknesses',[]))}W {len(swot.get('opportunities',[]))}O {len(swot.get('threats',[]))}T",
        swot.get("strategic_priority")))

    return {**state, "swot": swot, "events": events}


# ── Node 6: Report Writer ──────────────────────────────────────────────────────

async def report_writer(state: PipelineState) -> PipelineState:
    events = list(state.get("events", []))
    events.append(_evt("report_writer", "Writing final structured intelligence report"))

    sources = list({f["url"]: f for f in state["web_findings"] if f.get("url")}.values())[:10]

    llm = _llm(temperature=0.2)
    resp = await llm.ainvoke([
        SystemMessage(content="You are a senior market intelligence analyst writing for investors and founders. Output ONLY valid JSON."),
        HumanMessage(content=f"""
Business idea: "{state['idea']}"

You have the following research:
- Competitors: {json.dumps(state['competitor_findings'])[:1200]}
- Market sizing: {json.dumps(state['market_sizing'])[:800]}
- SWOT: {json.dumps(state['swot'])[:1200]}

Write a comprehensive market intelligence report.

Return JSON:
{{
  "executive_summary": "3-4 sentence verdict on this market opportunity. Be direct and honest.",
  "market_overview": "2-3 paragraph narrative covering market context, dynamics, and current state.",
  "opportunity_verdict": "go|proceed-with-caution|avoid",
  "opportunity_rationale": "One crisp sentence explaining the verdict.",
  "target_customer": {{
    "primary": "...",
    "demographics": "...",
    "psychographics": "...",
    "pain_points": ["...", "..."],
    "willingness_to_pay": "..."
  }},
  "go_to_market": {{
    "recommended_channel": "...",
    "launch_geography": "...",
    "pricing_strategy": "...",
    "early_milestones": ["Month 1-3: ...", "Month 4-6: ...", "Month 7-12: ..."]
  }},
  "investment_required": {{
    "minimum_viable": "...",
    "recommended": "...",
    "breakdown": {{"product": "...", "marketing": "...", "operations": "...", "team": "..."}}
  }},
  "key_risks": [
    {{"risk": "...", "mitigation": "..."}}
  ],
  "recommendations": ["...", "...", "..."]
}}
""")
    ])

    try:
        raw = resp.content.strip().replace("```json", "").replace("```", "")
        report = json.loads(raw)
    except Exception:
        report = {"executive_summary": "Report generation encountered an error.", "opportunity_verdict": "proceed-with-caution"}

    report["sources"] = [{"title": s.get("title", ""), "url": s.get("url", "")} for s in sources]
    report["idea"] = state["idea"]

    events.append(_evt("report_writer", "Report complete ✓"))
    return {**state, "report": report, "events": events}


# ── Graph Assembly ─────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("query_planner", query_planner)
    g.add_node("web_researcher", web_researcher)
    g.add_node("competitor_intel", competitor_intel)
    g.add_node("market_sizing", market_sizing)
    g.add_node("swot_synthesiser", swot_synthesiser)
    g.add_node("report_writer", report_writer)

    g.set_entry_point("query_planner")
    g.add_edge("query_planner", "web_researcher")
    g.add_edge("web_researcher", "competitor_intel")
    g.add_edge("competitor_intel", "market_sizing")
    g.add_edge("market_sizing", "swot_synthesiser")
    g.add_edge("swot_synthesiser", "report_writer")
    g.add_edge("report_writer", END)

    return g.compile()


GRAPH = build_graph()


async def stream_analysis(idea: str) -> AsyncIterator[str]:
    """Yield SSE-formatted strings as each node completes."""
    state: PipelineState = {
        "idea": idea,
        "queries": [],
        "web_findings": [],
        "competitor_findings": {},
        "market_sizing": {},
        "swot": {},
        "report": {},
        "events": [],
    }

    seen_events = 0
    async for chunk in GRAPH.astream(state):
        for node_name, node_state in chunk.items():
            events = node_state.get("events", [])
            new_events = events[seen_events:]
            for evt in new_events:
                yield f"data: {json.dumps(evt)}\n\n"
                await asyncio.sleep(0.05)
            seen_events = len(events)

            # If this is the report_writer node, emit the final report
            if node_name == "report_writer" and node_state.get("report"):
                yield f"data: {json.dumps({'node': 'done', 'msg': 'Analysis complete', 'data': node_state['report']})}\n\n"
