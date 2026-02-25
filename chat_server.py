"""
chat_server.py

Drop this file into: Kavah2026-main/Kavah2026-main/

Run with:
    pip install fastapi uvicorn
    python chat_server.py

Endpoints:
    POST /chat          { "message": "..." }  → { response, cris, latency_ms, turn }
    GET  /metrics       → current CRIS metrics
    POST /reset         → clear conversation history
    GET  /health        → { status: "ok" }
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from kavah_llm_bridge import KavahChat

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Kavah CRIS Chat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared agent instance ─────────────────────────────────────────────────────

agent = KavahChat(latent_dim=16, persistence_dir="memory")

# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "agent": "kavah-bedrock-agi"}

@app.post("/chat")
def chat(req: ChatRequest):
    return agent.chat(req.message)

@app.get("/metrics")
def metrics():
    raw = agent.agent.cris_monitor.metrics()
    return raw

@app.post("/reset")
def reset():
    agent.reset()
    return {"status": "conversation cleared"}

@app.get("/")
def index():
    """Serve the chat UI."""
    ui_path = os.path.join(os.path.dirname(__file__), "chat_ui.html")
    return FileResponse(ui_path)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "─" * 60)
    print("  KAVAH CRIS CHAT SERVER")
    print("  Open http://localhost:8000 in your browser")
    print("─" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")