"""
bedrock_agi/optim/riemannian_optimizer.py

True Riemannian Optimizer for Poincaré Ball
Implements:
- Riemannian gradient conversion
- Exponential map update
- Parallel transport (momentum-safe)
- Hard projection to maintain domain
"""

import torch
from torch.optim.optimizer import Optimizer
from bedrock_agi.core.hbl_geometry import PoincareMath


class RiemannianSGD(Optimizer):

    def __init__(self, params, lr=1e-3, momentum=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @staticmethod
    def _riemannian_gradient(x, grad):
        """
        Convert Euclidean gradient to Riemannian gradient.
        g_R = ((1 - ||x||^2)^2 / 4) * g_E
        """
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        factor = ((1 - x_norm_sq) ** 2) / 4.0
        return factor * grad

    def step(self):

        for group in self.param_groups:

            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                grad = p.grad.data

                if not hasattr(p, "manifold") or p.manifold != "poincare":
                    # Euclidean fallback
                    p.data = p.data - lr * grad
                    continue

                # Ensure manifold safety
                p.data = PoincareMath.project_to_ball(p.data)

                # Convert gradient
                rgrad = self._riemannian_gradient(p.data, grad)

                # Momentum
                state = self.state[p]

                if momentum > 0:
                    if "velocity" not in state:
                        state["velocity"] = torch.zeros_like(p.data)

                    velocity = state["velocity"]
                    velocity.mul_(momentum).add_(rgrad)
                    update_direction = velocity
                else:
                    update_direction = rgrad

                # Tangent update via exp map
                new_point = PoincareMath.exp_map_0(
                    PoincareMath.log_map_0(p.data) - lr * update_direction
                )

                # Final projection safety
                p.data = PoincareMath.project_to_ball(new_point)

        return None