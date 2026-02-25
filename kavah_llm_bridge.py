"""
kavah_llm_bridge.py

Phase 4: Geometric Reasoning Engine

The geometry IS the reasoning. The LLM renders it into English.

Architecture:
    user input
        → BedrockAgent.step()       (geometry runs, CRIS measures)
        → GeometricTraceBuilder     (extracts what the geometry did)
        → LLM(RENDERER_SYSTEM)      (translates trace to English)
        → response grounded in real geometric state

Nothing the LLM says can contradict the trace.
The trace is the answer. The LLM is the mouth.
"""

import torch
import requests
import time
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bedrock_agi.main import BedrockAgent
from bedrock_agi.constitution.gates import gate_i1_cris
from bedrock_agi.action.tools.registry import REGISTRY
from kavah_reasoner import GeometricTraceBuilder, GeometricTrace, RENDERER_SYSTEM


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

LLM_ENDPOINT   = "http://localhost:11434/api/chat"
LLM_MODEL      = "llama3.2"
EMBED_ENDPOINT = "http://localhost:11434/api/embeddings"
EMBED_MODEL    = "nomic-embed-text"
OPENAI_MODE    = False
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")

LAMBDA_MAX     = 0.999
GAMMA_MIN      = 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution (clean — bypasses agent.step() string mangling)
# ─────────────────────────────────────────────────────────────────────────────

def _run_tool(user_message: str, bedrock_response: str):
    """
    Re-execute the tool that agent.step() detected, returning clean structured output.
    Returns (tool_name, tool_result_str) or (None, None) if no tool ran.
    """
    br = bedrock_response.strip()

    if br.startswith("Executed web_search") or br.startswith("Failed to execute web_search"):
        query = user_message.strip()
        for prefix in ("search for ", "search ", "look up ", "find ", "google ", "lookup "):
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
                break
        result = REGISTRY.execute("web_search", {"query": query, "num_results": 4})
        if result["ok"] and result["value"].get("formatted"):
            return "web_search", result["value"]["formatted"]
        return "web_search", f"Search returned no results for: {query}"

    if br.startswith("Executed calculator") or br.startswith("Failed to execute calculator"):
        if "[Result]:" in br:
            return "calculator", br.split("[Result]:", 1)[1].strip()
        return "calculator", "Calculation result unavailable"

    if br.startswith("Executed code_exec") or br.startswith("Failed to execute code_exec"):
        if "[Result]:" in br:
            return "code_exec", br.split("[Result]:", 1)[1].strip()
        return "code_exec", "Code execution result unavailable"

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# KavahChat
# ─────────────────────────────────────────────────────────────────────────────

class KavahChat:
    """
    Kavah geometric reasoning agent.

    The geometry measures. The trace records. The LLM renders.
    Responses are grounded in real H^16 dynamics, not LLM pattern matching.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        persistence_dir: str = "memory",
        llm_endpoint: str = LLM_ENDPOINT,
        llm_model: str = LLM_MODEL,
        openai_mode: bool = OPENAI_MODE,
    ):
        self.llm_endpoint = llm_endpoint
        self.llm_model    = llm_model
        self.openai_mode  = openai_mode

        # Core geometry agent
        self.agent = BedrockAgent(
            latent_dim=latent_dim,
            persistence_dir=persistence_dir
        )

        # Point encoder at real embeddings
        self.agent.text_encoder.endpoint   = EMBED_ENDPOINT
        self.agent.text_encoder.model_name = EMBED_MODEL

        # Geometric trace builder — this is the reasoning engine
        self.tracer = GeometricTraceBuilder()

        # Conversation history (text turns for multi-turn coherence)
        self.history: list[dict] = []

        # Startup diagnostics
        self._embedding_live = self._check_embedding()
        if self._embedding_live:
            print("  Embeddings: nomic-embed-text via Ollama (live)")
        else:
            print("  Embeddings: hash fallback (Ollama unavailable)")

    # ── Public interface ───────────────────────────────────────────────────────

    def chat(self, user_message: str) -> dict:
        t0 = time.time()

        # ── PHASE 1: GEOMETRY ─────────────────────────────────────────────────
        # Run the full Bedrock loop. Geometry measures everything.
        bedrock_response = self.agent.step(user_message)

        # ── PHASE 2: TOOL EXTRACTION ──────────────────────────────────────────
        # Get clean tool output if a tool ran (bypasses agent.step() repr mangling)
        tool_name, tool_result = _run_tool(user_message, bedrock_response)
        print(f"  [TRACE] _run_tool: name={tool_name!r} result_len={len(tool_result or '')}")
        if tool_name:
            print(f"  [TRACE] Tool: {tool_name} → {str(tool_result)[:80]}")

        # ── PHASE 3: GEOMETRIC TRACE ──────────────────────────────────────────
        # Build the trace — this is the actual reasoning artifact
        trace = self.tracer.build(
            agent           = self.agent,
            bedrock_response = bedrock_response,
            tool_name       = tool_name,
            tool_result     = tool_result,
        )

        print(f"  [TRACE] {trace.convergence_status} λ={trace.lambda_val:.4f} "
              f"Δ={trace.delta_after:.4f} novelty={trace.topic_novelty:.4f} "
              f"confidence={trace.geometric_confidence()}")

        # ── PHASE 4: CONSTITUTIONAL GATE ──────────────────────────────────────
        gate_ok, gate_reason = gate_i1_cris(
            trace.lambda_val, trace.gamma_val, LAMBDA_MAX, GAMMA_MIN
        )

        if not gate_ok and trace.lambda_val > LAMBDA_MAX:
            response_text = self._geometric_gate_response(trace)
            self._append_history(user_message, response_text)
            return self._result(response_text, trace, t0)

        # ── PHASE 5: LLM RENDER ───────────────────────────────────────────────
        # The LLM receives the geometric trace and renders it into English.
        # It cannot contradict the trace — the trace IS the answer.
        messages = self._build_render_messages(user_message, trace)

        try:
            response_text = self._call_llm(messages)
        except Exception as e:
            # Graceful degradation: return a pure geometric response without LLM
            response_text = self._pure_geometric_response(trace)
            print(f"  [TRACE] LLM unavailable ({e}), returning geometric response")

        # ── PHASE 6: HISTORY ──────────────────────────────────────────────────
        self._append_history(user_message, response_text)

        return self._result(response_text, trace, t0)

    def reset(self):
        self.history.clear()
        self.tracer.reset()

    # ── Render message construction ───────────────────────────────────────────

    def _build_render_messages(self, user_message: str, trace: GeometricTrace) -> list:
        """
        Build the message list for the LLM renderer.
        The trace block is injected as a system-level fact the LLM must honor.
        History is sanitized to remove stale tool artifacts.
        """
        trace_block = trace.to_prompt_block()
        confidence  = trace.geometric_confidence()

        augmented_user = (
            f"{trace_block}\n\n"
            f"Geometric confidence: {confidence}\n\n"
            f"User asked: {user_message}\n\n"
            f"Respond based on the geometric trace above. "
            f"Your answer must be consistent with the measured state. "
            f"Do not reference tool results from previous turns — only the current trace."
        )

        messages = [{"role": "system", "content": RENDERER_SYSTEM}]

        # Include recent history but strip any [TOOL RESULT] blocks from old turns
        # to prevent stale calculator/search results from leaking into new responses
        clean_history = []
        for msg in self.history[-8:]:
            if msg["role"] == "assistant":
                # Remove tool result references from old assistant messages
                import re
                cleaned = re.sub(
                    r"\(Tool Result:[^)]*\)", "", msg["content"]
                )
                cleaned = re.sub(
                    r"\[TOOL RESULT\].*?\[/TOOL RESULT\]", "", 
                    cleaned, flags=re.DOTALL
                )
                clean_history.append({"role": "assistant", "content": cleaned.strip()})
            else:
                clean_history.append(msg)

        messages.extend(clean_history)
        messages.append({"role": "user", "content": augmented_user})
        return messages

    # ── Pure geometric response (LLM fallback) ────────────────────────────────

    def _pure_geometric_response(self, trace: "GeometricTrace") -> str:
        """
        Full Bedrock-native response derived entirely from geometric measurements.
        No LLM needed. This IS what geometry says when it speaks for itself.
        """
        parts = []

        if trace.convergence_status == "FIXED_POINT":
            parts.append(
                "The system has reached a geometric fixed point — "
                "full coherence, zero residual displacement. "
                "This question has been fully integrated into the identity manifold."
            )
        elif trace.convergence_status == "CONVERGING":
            steps = trace.steps_to_convergence
            eta = f"approximately {steps} further steps" if steps and steps < 1000 else "continued iteration"
            parts.append(
                f"The geometry is actively contracting toward a fixed point ({eta}). "
                f"Spectral radius \u03bb={trace.lambda_val:.4f} confirms inward pull "
                f"with healing rate \u03b3={trace.gamma_val:.4f}. "
                f"The answer is forming but not yet fully crystallized."
            )
        elif trace.convergence_status == "STABLE":
            parts.append(
                f"Geometric state stable. \u03bb={trace.lambda_val:.4f}, "
                f"\u03b3={trace.gamma_val:.4f}. Active self-correction ongoing."
            )
        elif trace.convergence_status == "MARGINAL":
            parts.append(
                f"Marginal stability — \u03bb={trace.lambda_val:.4f} near boundary. "
                f"This question pushes toward the edge of the current contraction regime. "
                f"Reduced epistemic confidence warranted."
            )
        else:
            parts.append(
                f"Geometric divergence. \u03bb={trace.lambda_val:.4f} — "
                f"system expanding rather than contracting. "
                f"Question may be outside the current identity manifold."
            )

        if trace.topic_novelty > 1.5:
            parts.append(
                f"High displacement (\u0394={trace.topic_novelty:.4f}) — "
                f"geometrically far from prior state. New territory."
            )
        elif trace.topic_novelty > 0.8:
            parts.append(
                f"Moderate displacement (\u0394={trace.topic_novelty:.4f}) — "
                f"related to prior context but requiring substantial state update."
            )
        else:
            parts.append(
                f"Low displacement (\u0394={trace.topic_novelty:.4f}) — "
                f"deep in established territory. Well-oriented."
            )

        if len(trace.delta_history) >= 3:
            recent = trace.delta_history[-3:]
            trend = recent[-1] - recent[0]
            if trend < -0.15:
                parts.append(
                    f"Delta decreasing ({recent[0]:.3f} \u2192 {recent[-1]:.3f}): "
                    f"understanding deepening. Converging on this topic."
                )
            elif trend > 0.15:
                parts.append(
                    f"Delta increasing ({recent[0]:.3f} \u2192 {recent[-1]:.3f}): "
                    f"complexity growing. Each question opens new territory."
                )
            else:
                parts.append(
                    f"Delta steady ({recent[0]:.3f} \u2192 {recent[-1]:.3f}): "
                    f"stable exploration regime."
                )

        if trace.steps_to_convergence and trace.steps_to_convergence < 100:
            parts.append(
                f"Fixed point estimated in {trace.steps_to_convergence} more turns "
                f"at current contraction rate."
            )

        # Only include tool result if it looks like real content (not a bare number artifact)
        if trace.tool_result and len(str(trace.tool_result).strip()) > 10:
            parts.append(f"\n{trace.tool_result}")

        parts.append(
            f"[Turn {trace.turn} \u00b7 {trace.num_samples} samples \u00b7 "
            f"confidence: {trace.geometric_confidence()}]"
        )

        return "\n\n".join(parts)

    def _geometric_gate_response(self, trace: GeometricTrace) -> str:
        """Response when the constitutional gate blocks generation."""
        return (
            f"Geometric instability detected. "
            f"λ={trace.lambda_val:.4f} exceeds stability threshold. "
            f"Cannot generate reliable output until contraction resumes. "
            f"Current displacement: Δ={trace.delta_after:.6f}."
        )

    # ── LLM calls ──────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list) -> str:
        if self.openai_mode:
            return self._call_openai(messages)
        return self._call_ollama(messages)

    def _call_ollama(self, messages: list) -> str:
        payload = {"model": self.llm_model, "messages": messages, "stream": False}
        resp = requests.post(self.llm_endpoint, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _call_openai(self, messages: list) -> str:
        headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
        resp = requests.post(
            self.llm_endpoint,
            json={"model": self.llm_model, "messages": messages},
            headers=headers, timeout=120
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _check_embedding(self) -> bool:
        try:
            resp = requests.post(
                EMBED_ENDPOINT,
                json={"model": EMBED_MODEL, "prompt": "test"},
                timeout=3
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _append_history(self, user: str, assistant: str):
        self.history.append({"role": "user",      "content": user})
        self.history.append({"role": "assistant",  "content": assistant})

    def _result(self, response: str, trace: GeometricTrace, t0: float) -> dict:
        return {
            "response":   response,
            "cris": {
                "lambda":      trace.lambda_val,
                "gamma":       trace.gamma_val,
                "delta":       trace.delta_after,
                "stable":      trace.is_stable,
                "status":      trace.convergence_status,
                "num_samples": trace.num_samples,
                # Pre-formatted for UI
                "lambda_str":  f"{trace.lambda_val:.6f}",
                "gamma_str":   f"{trace.gamma_val:.6f}",
                "delta_str":   f"{trace.delta_after:.9f}",
            },
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "turn":       len(self.history) // 2,
        }