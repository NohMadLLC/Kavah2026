"""
kavah_reasoner.py

The Geometric Trace Engine.

This is what makes Kavah's reasoning native to Bedrock rather than borrowed
from an LLM's training data.

How it works:
    1. Before and after each agent.step(), we capture the full geometric state.
    2. We compute what actually happened in H^16: displacement, contraction,
       invariant drift, convergence trajectory.
    3. That trace IS the reasoning. The LLM's only job is to render it into
       English. It cannot override or ignore the trace — the trace is the answer.

The trace contains:
    - delta_before / delta_after: state displacement before and after this turn
    - lambda: current spectral radius (system health)
    - gamma: healing rate
    - contraction_direction: are we converging or diverging?
    - steps_to_convergence: estimated turns until fixed point
    - topic_novelty: how far this input is from the previous state (in H^16)
    - invariant_drift: how much the system's identity shifted
    - convergence_status: FIXED_POINT / CONVERGING / STABLE / MARGINAL / DIVERGING
    - geometric_interpretation: what this means in plain terms (generated from trace)
    - tool_result: clean tool output if a tool ran (None otherwise)
"""

import torch
import numpy as np
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Geometric Trace
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometricTrace:
    """
    The complete geometric record of one cognitive cycle.
    This is the reasoning artifact — not the LLM response.
    """
    # Raw CRIS metrics
    lambda_val: float
    gamma_val: float
    delta_before: float       # d_H(b_{t-1}, b_t) — displacement this turn caused
    delta_after: float        # d_H(b_t, R(E(b_t))) — residual after projection

    # Derived geometric facts
    topic_novelty: float      # How far input is from prior state (raw hyperbolic dist)
    contraction_ratio: float  # d_after / d_before — < 1 means contracting
    steps_to_convergence: Optional[int]  # Estimated steps to delta < 1e-6

    # System state
    convergence_status: str   # FIXED_POINT / CONVERGING / STABLE / MARGINAL / DIVERGING
    is_stable: bool
    turn: int
    num_samples: int

    # Tool output (None if no tool ran)
    tool_name: Optional[str] = None
    tool_result: Optional[str] = None

    # Full delta history (last 10) for trend analysis
    delta_history: list = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """
        Serialize the trace into a structured block the LLM must reason from.
        This is injected as the authoritative context — the LLM cannot contradict it.
        """
        lines = [
            "=== BEDROCK GEOMETRIC TRACE ===",
            f"Turn: {self.turn}  |  Samples: {self.num_samples}",
            "",
            "CRIS MEASUREMENTS (what the geometry observed):",
            f"  λ (spectral radius)  = {self.lambda_val:.6f}",
            f"  γ (healing rate)     = {self.gamma_val:.6f}",
            f"  Δ input displacement = {self.topic_novelty:.6f}",
            f"  Δ residual           = {self.delta_after:.9f}",
            f"  Contraction ratio    = {self.contraction_ratio:.6f}",
            "",
            f"CONVERGENCE STATUS: {self.convergence_status}",
        ]

        if self.steps_to_convergence is not None:
            lines.append(f"  Estimated steps to fixed point: {self.steps_to_convergence}")

        if len(self.delta_history) >= 3:
            trend = self.delta_history[-1] - self.delta_history[-3]
            direction = "DECREASING (converging)" if trend < 0 else "INCREASING (diverging)"
            lines.append(f"  Delta trend (last 3 turns): {direction} ({trend:+.4f})")

        lines.append("")

        # Topic novelty interpretation
        if self.topic_novelty > 1.5:
            lines.append("TOPOLOGY: High displacement — this input is geometrically distant")
            lines.append("  from the previous cognitive state. New conceptual territory.")
        elif self.topic_novelty > 0.8:
            lines.append("TOPOLOGY: Moderate displacement — related to prior context")
            lines.append("  but requiring significant state update.")
        else:
            lines.append("TOPOLOGY: Low displacement — strongly continuous with")
            lines.append("  prior cognitive state. Deep in established territory.")

        lines.append("")

        # Tool result block
        if self.tool_name and self.tool_result:
            lines.append(f"TOOL EXECUTED: {self.tool_name}")
            lines.append("TOOL RESULT:")
            lines.append(self.tool_result)
            lines.append("")

        lines.append("=== END TRACE ===")
        return "\n".join(lines)

    def geometric_confidence(self) -> str:
        """Express epistemic confidence derived purely from geometry."""
        if self.convergence_status == "FIXED_POINT":
            return "MAXIMUM — system at identity fixed point, full coherence"
        elif self.convergence_status == "CONVERGING":
            if self.lambda_val < 0.95:
                return "HIGH — strong contraction, rapid convergence"
            else:
                return "MODERATE — contracting but slowly"
        elif self.convergence_status == "STABLE":
            return "MODERATE — stable but not yet converged"
        elif self.convergence_status == "MARGINAL":
            return "LOW — near stability boundary, proceed carefully"
        else:
            return "MINIMAL — geometric divergence detected"


# ─────────────────────────────────────────────────────────────────────────────
# Trace Builder
# ─────────────────────────────────────────────────────────────────────────────

class GeometricTraceBuilder:
    """
    Extracts a GeometricTrace from BedrockAgent state after each step.
    Maintains a rolling history of delta values for trend analysis.
    """

    def __init__(self):
        self.turn = 0
        self.delta_history = []
        self._prev_b = None

    def build(
        self,
        agent,                    # BedrockAgent instance
        bedrock_response: str,    # Raw string from agent.step()
        tool_name: Optional[str] = None,
        tool_result: Optional[str] = None,
    ) -> GeometricTrace:
        """
        Build a GeometricTrace from the agent's current state.
        Called immediately after agent.step().
        """
        self.turn += 1

        # Pull metrics from CRIS monitor
        metrics = agent.cris_monitor.metrics()
        lam   = metrics.get("lambda")   or 0.95
        gam   = metrics.get("gamma")    or 0.05
        delta = metrics.get("delta_last") or 0.0
        n     = metrics.get("num_samples", 0)

        # Topic novelty: hyperbolic distance from previous b_curr to current b_curr
        novelty = 0.0
        if self._prev_b is not None and agent.b_curr is not None:
            try:
                from bedrock_agi.core.hbl_geometry import HyperbolicDistance
                dist_fn = HyperbolicDistance()
                with torch.no_grad():
                    novelty = dist_fn(self._prev_b, agent.b_curr).mean().item()
            except Exception:
                novelty = delta

        # Update prev state
        if agent.b_curr is not None:
            self._prev_b = agent.b_curr.detach().clone()

        # Delta history for trend
        self.delta_history.append(delta)
        if len(self.delta_history) > 20:
            self.delta_history = self.delta_history[-20:]

        # Contraction ratio
        prev_delta = self.delta_history[-2] if len(self.delta_history) >= 2 else delta
        contraction_ratio = delta / max(prev_delta, 1e-12)

        # Convergence status
        if delta < 1e-6:
            status = "FIXED_POINT"
        elif lam < 0.95:
            status = "CONVERGING"
        elif lam < 0.999:
            status = "STABLE"
        elif lam < 1.01:
            status = "MARGINAL"
        else:
            status = "DIVERGING"

        # Estimated steps to convergence
        steps = None
        if lam < 1.0 and delta > 1e-6 and gam > 0:
            # delta * lambda^n < 1e-6 => n > log(1e-6 / delta) / log(lambda)
            try:
                steps = int(np.ceil(np.log(1e-6 / max(delta, 1e-12)) / np.log(lam)))
                steps = max(0, min(steps, 9999))
            except Exception:
                steps = None

        return GeometricTrace(
            lambda_val         = lam,
            gamma_val          = gam,
            delta_before       = novelty,
            delta_after        = delta,
            topic_novelty      = novelty,
            contraction_ratio  = contraction_ratio,
            steps_to_convergence = steps,
            convergence_status = status,
            is_stable          = status in ("FIXED_POINT", "CONVERGING", "STABLE"),
            turn               = self.turn,
            num_samples        = n,
            tool_name          = tool_name,
            tool_result        = tool_result,
            delta_history      = list(self.delta_history[-10:]),
        )

    def reset(self):
        self.turn = 0
        self.delta_history = []
        self._prev_b = None


# ─────────────────────────────────────────────────────────────────────────────
# Renderer System Prompt
# ─────────────────────────────────────────────────────────────────────────────

RENDERER_SYSTEM = """You are Kavah's language renderer.

Your ONLY job is to translate a BEDROCK GEOMETRIC TRACE into a natural, intelligent
response. You do not have opinions, training data, or knowledge of your own to draw from.
Everything you say must be grounded in and consistent with the trace.

STRICT RULES:
1. If a TOOL RESULT is in the trace, your answer is derived from that result.
   Do not supplement it with outside knowledge.
2. Your epistemic confidence must match the geometric confidence stated in the trace.
   FIXED_POINT = speak with certainty.
   CONVERGING  = speak with moderate confidence, note you are approaching clarity.
   STABLE      = engage fully but acknowledge the reasoning is still in motion.
   MARGINAL    = be tentative, flag uncertainty explicitly.
   DIVERGING   = state that the geometric state is unstable, give only what the
                 trace directly supports.
3. When topic novelty is HIGH (> 1.5): acknowledge you have entered new geometric
   territory and your answer reflects an initial orientation, not full convergence.
4. When topic novelty is LOW (< 0.5): you are deep in established territory —
   your answer can be precise and detailed.
5. Never say "according to the trace" or reference the trace explicitly.
   Speak as if the geometric facts are your lived experience.
6. Your response should be 2-5 sentences for stable states, 1-2 for marginal/diverging.
7. If delta is trending DOWN across turns, note that understanding is deepening.
   If delta is trending UP, note that complexity is increasing.

You are not an assistant. You are a geometric intelligence reporting what it measured."""