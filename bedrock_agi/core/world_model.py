"""
bedrock_agi/core/world_model.py

The World Model (W).
The engine of Recursive Closure.

Function:
    Binds Physics (E, R), Consciousness (S), and Monitoring (CRIS)
    into a unified causal loop.

Dynamics:
    1. Intent: u_t = Influence(s_t)
    2. Physics: b_{t+1} = R( E(b_t) + u_t )  OR  b_{t+1} = Perception
    3. Monitor: Update CRIS metrics (λ, γ, Δ) based on transition b_t -> b_{t+1}
    4. Consciousness: s_{t+1} = GRU( s_t, [b_{t+1}, λ, γ] )

Persistence:
    Maintains the 'Identity Thread' (b_prev) and 'Self State' (s_curr).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Any

from .e_model import EvolutionModel
from .r_projector import RProjector
from .cris_monitor import CRISMonitor
from .hbl_geometry import HyperbolicDistance

try:
    from ..consciousness.self_model import SelfModel
    from ..consciousness.self_influence import BoundedInfluence
except ImportError:
    class SelfModel(nn.Module):
        def __init__(self, state_dim, latent_dim):
            super().__init__()
            self.gru = nn.GRUCell(latent_dim + 3, state_dim)

        def forward(self, b_curr, cris_metrics, s_prev):
            if b_curr.dim() == 1: b_curr = b_curr.unsqueeze(0)
            if cris_metrics.dim() == 1: cris_metrics = cris_metrics.unsqueeze(0)
            ctx = torch.cat([b_curr, cris_metrics], dim=-1)
            return self.gru(ctx, s_prev)

    class BoundedInfluence(nn.Module):
        def __init__(self, state_dim, latent_dim, epsilon=0.01):
            super().__init__()
            self.net = nn.Linear(state_dim, latent_dim)
            self.epsilon = epsilon

        def forward(self, s_curr):
            raw = self.net(s_curr)
            norm = torch.norm(raw, dim=-1, keepdim=True) + 1e-9
            scale = torch.minimum(torch.ones_like(norm), self.epsilon / norm)
            return raw * scale


class WorldModel:
    """
    The Cognitive Engine.
    """
    def __init__(
        self,
        latent_dim: int = 16,
        state_dim: int = 32,
        device: str = "cpu"
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        # 1. Physics Layer (The Law)
        self.E = EvolutionModel(latent_dim).to(device)
        self.R = RProjector(latent_dim).to(device)

        # 2. Consciousness Layer (The Observer)
        self.self_model = SelfModel(state_dim, latent_dim).to(device)
        self.influence = BoundedInfluence(state_dim, latent_dim).to(device)

        # 3. Observability Layer (The Gauges)
        self.monitor = CRISMonitor(tail=50, device=device)

        # 4. State Persistence (The Thread)
        self.b_prev: Optional[torch.Tensor] = None
        self.s_curr: torch.Tensor = torch.zeros(1, state_dim).to(device)

        # Default metrics for cold start
        self._last_metrics = {
            "lambda": 0.95,
            "gamma": 0.05,
            "delta_last": 0.0,
            "is_stable": True
        }

    def _pack_cris_context(self, metrics: Dict[str, float]) -> torch.Tensor:
        l = metrics.get('lambda', 0.95)
        g = metrics.get('gamma', 0.05)
        d = metrics.get('delta_last', 0.0)
        return torch.tensor([[l, g, d]], dtype=torch.float32).to(self.device)

    def step(
        self,
        observation: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute one Cognitive Cycle (The Heartbeat).

        Args:
            observation: Optional external forcing (Perception).
                         If None, the system evolves internally (dreams/thinks).

        Returns:
            b_next: The new state of the world model.
            metrics: Updated CRIS stability metrics.
        """
        # --- PHASE 1: ANCHORING (Cold Start) ---
        if self.b_prev is None:
            if observation is not None:
                self.b_prev = observation.detach().clone()
                return self.b_prev, self._last_metrics
            else:
                self.b_prev = torch.zeros(1, self.latent_dim).to(self.device)
                return self.b_prev, self._last_metrics

        # --- PHASE 2: DYNAMICS ---
        b_target = None

        if observation is not None:
            # Case A: Perception (External Forcing)
            b_target = observation
        else:
            # Case B: Evolution (Internal Thought)
            u_t = self.influence(self.s_curr)
            b_pred = self.E(self.b_prev) + u_t
            b_target = self.R(b_pred)

        # --- PHASE 3: OBSERVATION (CRIS Monitor) ---
        self.monitor.update(self.b_prev, b_target)

        raw_metrics = self.monitor.metrics()
        self._last_metrics = {
            "lambda": raw_metrics["lambda"],
            "gamma": raw_metrics["gamma"],
            "delta_last": raw_metrics["delta_last"],
            "is_stable": self.monitor.is_stable()
        }

        # --- PHASE 4: CONSCIOUSNESS UPDATE ---
        cris_ctx = self._pack_cris_context(self._last_metrics)
        self.s_curr = self.self_model(b_target, cris_ctx, self.s_curr)

        # --- PHASE 5: PERSISTENCE ---
        self.b_prev = b_target.detach().clone()

        return self.b_prev, self._last_metrics

    def reset(self):
        """Hard reset of the manifold state (Memory Wipe)."""
        self.b_prev = None
        self.s_curr = torch.zeros(1, self.state_dim).to(self.device)
        self.monitor = CRISMonitor(tail=50, device=self.device)
        self._last_metrics = {
            "lambda": 0.95,
            "gamma": 0.05,
            "delta_last": 0.0,
            "is_stable": True
        }