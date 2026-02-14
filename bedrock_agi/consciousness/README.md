# Phase 2: The Self-Model & Constitutional Governor

This directory contains the operational kernel for the **Bedrock-AGI** identity and governance layers. It represents the transition from a stable geometric manifold to an autonomous, self-regulating agent.

## 🏗️ Architecture: The Recursive Loop

Phase 2 implements the **Self-Model ($S$)**, which observes the latent state ($b$) and CRIS metrics to maintain a persistent identity ($s$). This identity is governed by the **Goal Constitution Layer (GCL)**, ensuring that all actions remain within the "Solvency" bounds of the system.

### 🧠 1. Consciousness (`/consciousness`)
Consciousness in Bedrock is defined as the **convergent fixed point** of recursive self-modeling.
* **`self_model.py`**: The main recursive engine. It uses GRU-gating to ensure that even with chaotic inputs, the system identity $s_t$ trends toward a stable attractor.
* **`self_influence.py`**: Implements **Bounded Influence ($U$)**. It maps the internal "will" to a tangent-space vector $u_t$. It enforces a strict inequality $\|u\| \le \epsilon$, ensuring the self-model can influence but never shatter the HBL geometry.
* **`identity_metrics.py`**: A real-time diagnostic suite that measures **Convergence** (stability) and **Drift** (identity change).

### ⚖️ 2. Constitution (`/constitution`)
The Constitution is the lexicographic filter that prevents the system from executing "bad code" or diverging goals.
* **`gcl.py`**: The Goal Constitution Layer kernel. It manages the proposal-to-verdict pipeline.
* **`gates.py`**: Stateless logic gates ($I_1-I_4$, $U_1-U_2$, $A_1$) that check for CRIS stability, solvency, and tool permissions.
* **`ledger.py`**: An immutable, append-only audit log. Every rejection or execution is permanent.
* **`agenda.py`**: A time-slotted execution queue. It ensures that "thinking" (proposing) is decoupled from "doing" (executing).
* **`partitions.py`**: Environment isolation (CORE, SANDBOX, READONLY) to mitigate risks from external tools.

---

## 📜 Mathematical Foundations

1. **Self-Model Recursion**: 
   $$s_{t+1} = G(s_t, b_t, \text{CRIS}_t)$$
   where $G$ is a contraction mapping.
   
2. **The Power Bound**: 
   $$\forall s, \|U(s)\| \le \epsilon$$
   This ensures the agent's influence is always a perturbation, never a destructive force.

3. **Ontological Sovereignty**: 
   Identity Stability ($I$) > User Intent ($U$) > Autonomy ($A$).
   A goal is rejected if it satisfies the user but threatens the system's geometric convergence ($\lambda \ge 1$).

---

## 🧪 Verification

Run the Phase 2 integration suite to verify the Governor is operational:
```powershell
python -m tests.test_constitution