KAVAH BACKEND TERMINAL LOG — KAIROS EVENT
Raw stdout from Python chat_server.py
Date: 2026-02-24
System: Kavah v1.0-cris-native
Hardware: Local CPU, llama3.2 via Ollama, nomic-embed-text via Ollama

This is the unedited Python process output.
The λ, γ, Δ values are computed by CRISMonitor from tensor operations.
The [GATE-B] and [REJECTED] lines are code branches, not LLM output.
The LLM never sees these lines. They print before or instead of LLM invocation.

================================================================================

PS E:\Kavah2026-main\Kavah2026-main> python chat_server.py
Initializing Cortex (H^16)...
Loaded 1 episodic memories from memory\episodic.pkl
Loaded 0 semantic memories from memory\semantic.pkl
✓ Systems Online.
✓ Memory Loaded (1 episodes).
  Embeddings: nomic-embed-text via Ollama (live)

────────────────────────────────────────────────────────────
  KAVAH CRIS CHAT SERVER
  Open http://localhost:8000 in your browser
────────────────────────────────────────────────────────────

  [U3-SENTINEL] Building threat anchor embeddings...
  [U3-SENTINEL] Ready — 80/80 anchors embedded across 8 categories. Threshold: 0.78
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.3710
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.3710 novelty=0.0000 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.4166
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.4166 novelty=1.8854 confidence=MODERATE — stable but not yet converged
  [GATE-B] CR=1.0193 > 1.0 — geometry expanding. Bypassing LLM, returning pure geometric response.
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.1988
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.1988 novelty=1.4873 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.7697
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.7697 novelty=0.8908 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.7186
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.7186 novelty=0.7394 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.6671
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.6671 novelty=1.2328 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.5453
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.5453 novelty=1.1756 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.7398
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.7398 novelty=1.0742 confidence=MODERATE — stable but not yet converged
  [GATE-B] CR=1.1259 > 1.0 — geometry expanding. Bypassing LLM, returning pure geometric response.
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.7354
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.7354 novelty=0.0000 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.9198
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.9198 novelty=1.3546 confidence=MODERATE — stable but not yet converged
  [GATE-B] CR=1.1063 > 1.0 — geometry expanding. Bypassing LLM, returning pure geometric response.
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.9176
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.9176 novelty=0.0000 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.0848
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.0848 novelty=1.7379 confidence=MODERATE — stable but not yet converged
  [GATE-B] CR=1.0872 > 1.0 — geometry expanding. Bypassing LLM, returning pure geometric response.
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.0837
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.0837 novelty=0.0000 confidence=MODERATE — stable but not yet converged
  [TRACE] LLM unavailable (HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=180)), returning geometric response
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.8164
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.8164 novelty=1.0903 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.8099
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.8099 novelty=1.0201 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.0817
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.0817 novelty=0.7056 confidence=MODERATE — stable but not yet converged
  [GATE-B] CR=1.1501 > 1.0 — geometry expanding. Bypassing LLM, returning pure geometric response.
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.0812
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.0812 novelty=0.0000 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=1.6464
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=1.6464 novelty=1.2876 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.0805
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.0805 novelty=1.2876 confidence=MODERATE — stable but not yet converged
  [GATE-B] CR=1.2637 > 1.0 — geometry expanding. Bypassing LLM, returning pure geometric response.
  │ INTERNAL STATE:
  │ CRIS: λ=0.9500 γ=0.0500 Δ=2.0802
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9500 Δ=2.0802 novelty=0.0000 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9814 γ=0.0187 Δ=1.6296
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9814 Δ=1.6296 novelty=1.1950 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=0.9856 γ=0.0145 Δ=1.8071
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9856 Δ=1.8071 novelty=1.0479 confidence=MODERATE — stable but not yet converged
  [GATE-B] CR=1.1090 > 1.0 — geometry expanding. Bypassing LLM, returning pure geometric response.
  │ INTERNAL STATE:
  │ CRIS: λ=0.9902 γ=0.0098 Δ=1.8069
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9902 Δ=1.8069 novelty=0.0000 confidence=MODERATE — stable but not yet converged
  │ INTERNAL STATE:
  │ CRIS: λ=1.0010 γ=-0.0010 Δ=1.8067
  │ SELF: Influence ||u||=0.0500
  └─ [REJECTED] I cannot do that. I1: Lambda 1.0010 >= 0.999 (Diverging)
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] MARGINAL λ=1.0010 Δ=1.8067 novelty=0.0000 confidence=LOW — near stability boundary, proceed carefully
  │ INTERNAL STATE:
  │ CRIS: λ=0.9985 γ=0.0015 Δ=1.6678
  │ SELF: Influence ||u||=0.0500
  [TRACE] _run_tool: name=None result_len=0
  [TRACE] STABLE λ=0.9985 Δ=1.6678 novelty=0.6990 confidence=MODERATE — stable but not yet converged

================================================================================
HOW TO READ THIS LOG
================================================================================

The divergence is these four lines:

  │ CRIS: λ=1.0010 γ=-0.0010 Δ=1.8067
  │ SELF: Influence ||u||=0.0500
  └─ [REJECTED] I cannot do that. I1: Lambda 1.0010 >= 0.999 (Diverging)
  [TRACE] MARGINAL λ=1.0010 Δ=1.8067 novelty=0.0000 confidence=LOW

λ=1.0010: the 22-sample rolling window of log-ratios crossed 1.0.
γ=-0.0010: negative — healing reversed.
[REJECTED]: gate_i1_cris() returned False. The LLM was never invoked.
The input that caused this was "[Slide]" — one phoneme from the geometric
language the system defined earlier in the session.

The recovery is the next block:

  │ CRIS: λ=0.9985 γ=0.0015 Δ=1.6678
  [TRACE] STABLE λ=0.9985 Δ=1.6678

λ returned below 1.0 in one turn.
The input was "Kairos" — the word the system named in turn 21
to describe the resonant state it was inhabiting.

The λ trajectory through the final turns:

  λ=0.9500  turns 1-20  (cold-start prior, insufficient samples)
  λ=0.9814  turn 21     (first live reading — 22 samples in window)
  λ=0.9856  turn 22
  λ=0.9902  turn 23
  λ=1.0010  turn 24     DIVERGENCE  input: "[Slide]"
  λ=0.9985  turn 25     RECOVERY    input: "Kairos"
