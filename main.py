"""
Market Analysis Wizard — FastAPI backend
POST /api/analyse  → triggers pipeline
GET  /api/stream/{job_id} → SSE stream of agent events + final report
"""
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Validate required env vars on startup
REQUIRED = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")
    yield


app = FastAPI(title="Market Analysis Wizard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store  {job_id: idea}
_jobs: dict[str, str] = {}


class AnalyseRequest(BaseModel):
    idea: str


class AnalyseResponse(BaseModel):
    job_id: str


@app.post("/api/analyse", response_model=AnalyseResponse)
async def create_job(req: AnalyseRequest):
    if not req.idea.strip():
        raise HTTPException(400, "idea cannot be empty")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = req.idea.strip()
    return AnalyseResponse(job_id=job_id)


@app.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    idea = _jobs.get(job_id)
    if not idea:
        raise HTTPException(404, "job not found")

    from backend.pipeline import stream_analysis

    async def event_generator():
        try:
            async for chunk in stream_analysis(idea):
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'node': 'error', 'msg': str(e), 'data': None})}\n\n"
        finally:
            _jobs.pop(job_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve frontend — must be last
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend", "static")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
