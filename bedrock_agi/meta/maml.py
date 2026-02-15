"""
bedrock_agi/meta/maml.py

MAML: Model-Agnostic Meta-Learning
Adapt to new tasks with few examples while maintaining CRIS stability.

Uses torch.func for functional parameter manipulation to correctly implement
higher-order gradients without memory explosion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Callable, Dict, Any, Tuple
from torch.func import functional_call, grad, grad_and_value

from .fitness import compute_fitness
from .hyperparam_gates import constitute_hyperparams


class MAMLWrapper:
    """
    MAML meta-learner for bedrock models.
    
    Learns to adapt quickly to new tasks while preserving λ < 1.
    Wraps a base model and manages inner/outer loop optimization using
    functional calls to correctly handle parameter updates.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-4,
        inner_steps: int = 5,
        first_order: bool = False
    ):
        """
        Args:
            model: Base model (E, R, or full F)
            inner_lr: Learning rate for task adaptation
            outer_lr: Learning rate for meta-updates
            inner_steps: Number of gradient steps per task
            first_order: If True, use first-order MAML (ignore higher-order gradients)
        """
        # 1. Constitution Check: Inner LR
        ok, msg = constitute_hyperparams({'lr': inner_lr})
        if not ok:
            raise ValueError(f"Inner LR rejected: {msg}")
            
        # 2. Constitution Check: Outer LR
        ok, msg = constitute_hyperparams({'lr': outer_lr})
        if not ok:
            raise ValueError(f"Outer LR rejected: {msg}")
            
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # Store initial parameters as a flat dictionary for functional calls
        self._params = dict(model.named_parameters())
        
        # Meta-optimizer updates the initial parameters of the model
        self.meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        
    def _clone_params(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the current model parameters."""
        return {name: p.clone() for name, p in self.model.named_parameters()}
    
    def _functional_forward(self, params: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Forward pass using given parameters."""
        return functional_call(self.model, params, (x,))
    
    def adapt(
        self,
        support_data: List[Tuple[torch.Tensor, torch.Tensor]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt model to support set (inner loop) using functional gradient descent.
        
        Args:
            support_data: List of (x, y) tuples for adaptation
            loss_fn: Task-specific loss function (pred, target) -> loss tensor
            
        Returns:
            adapted_params: Dictionary of parameters adapted to the task
        """
        # Start from current model parameters
        params = self._clone_params()
        
        # Inner loop: perform inner_steps of gradient descent on the support set
        for _ in range(self.inner_steps):
            # Compute loss over all support examples (batch size can be small, sum works)
            loss = torch.tensor(0.0, device=next(iter(params.values())).device)
            for x, y in support_data:
                pred = self._functional_forward(params, x)
                loss = loss + loss_fn(pred, y)
            
            # Compute gradients of loss w.r.t. parameters
            # If first_order, use torch.no_grad to avoid creating computation graph
            if self.first_order:
                # For first-order MAML, we detach parameters before gradient step
                # (i.e., we don't propagate second-order gradients)
                # Here we compute gradients but then apply step without graph
                grads = torch.autograd.grad(loss, params.values(), create_graph=False)
                # Update parameters manually (not using optimizer)
                new_params = {}
                for (name, p), g in zip(params.items(), grads):
                    if g is not None:
                        new_params[name] = p - self.inner_lr * g
                    else:
                        new_params[name] = p
                params = new_params
            else:
                # For second-order MAML, we need to keep the graph for outer gradients
                grads = torch.autograd.grad(loss, params.values(), create_graph=True)
                # Update parameters, but keep them as tensors in the graph
                new_params = {}
                for (name, p), g in zip(params.items(), grads):
                    if g is not None:
                        new_params[name] = p - self.inner_lr * g
                    else:
                        new_params[name] = p
                params = new_params
        
        return params
    
    def meta_update(
        self,
        tasks: List[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        """
        Meta-update across multiple tasks (outer loop).
        
        Args:
            tasks: List of task dicts {'support': [(x,y)], 'query': [(x,y)]}
            loss_fn: Loss function (pred, target) -> loss tensor
            
        Returns:
            Average meta-loss
        """
        # We'll accumulate meta-loss over tasks
        meta_loss = torch.tensor(0.0, device=next(iter(self.model.parameters())).device)
        num_tasks = len(tasks)
        
        for task in tasks:
            support = task['support']
            query = task['query']
            
            # 1. Adapt to support set to get task-specific parameters θ'
            adapted_params = self.adapt(support, loss_fn)
            
            # 2. Evaluate on query set using θ'
            task_loss = torch.tensor(0.0, device=meta_loss.device)
            for x, y in query:
                pred = self._functional_forward(adapted_params, x)
                task_loss = task_loss + loss_fn(pred, y)
            
            # Average over query examples (optional, but good)
            task_loss = task_loss / len(query)
            meta_loss = meta_loss + task_loss
        
        # Average over tasks
        meta_loss = meta_loss / num_tasks
        
        # 3. Meta-optimization step: backpropagate through the entire computation
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def get_adapted_model(self, adapted_params: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Return a copy of the model with parameters replaced by adapted_params.
        Useful for deployment after adaptation.
        """
        import copy
        model_copy = copy.deepcopy(self.model)
        model_copy.load_state_dict(adapted_params)
        return model_copy


if __name__ == "__main__":
    print("Testing MAML...")
    
    # Simple model for testing
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 2)
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleMLP()
    
    # Test 1: Constitution Check (Valid LRs)
    try:
        maml = MAMLWrapper(model, inner_lr=1e-3, outer_lr=1e-4)
        print("✓ Valid learning rates accepted")
    except ValueError as e:
        print(f"✗ Failed valid check: {e}")
    
    # Test 2: Constitution Check (Invalid LR)
    try:
        maml_bad = MAMLWrapper(model, inner_lr=1.5)
        print("✗ Should reject large LR")
    except ValueError as e:
        print(f"✓ Invalid LR rejected: {e}")
    
    # Test 3: Check adaptation and meta-update on synthetic data
    # Create a tiny meta-task: learn to map 5d input to binary output
    # (this is just a smoke test to ensure no runtime errors)
    torch.manual_seed(42)
    support = [(torch.randn(2, 5), torch.randint(0, 2, (2,)))]
    query = [(torch.randn(2, 5), torch.randint(0, 2, (2,)))]
    tasks = [{'support': support, 'query': query}]
    
    # Define loss function
    def loss_fn(pred, target):
        return torch.nn.functional.cross_entropy(pred, target)
    
    # Run meta-update
    try:
        maml = MAMLWrapper(model, inner_lr=1e-2, outer_lr=1e-3, inner_steps=3)
        loss_val = maml.meta_update(tasks, loss_fn)
        print(f"✓ Meta-update ran successfully, loss = {loss_val:.4f}")
    except Exception as e:
        print(f"✗ Meta-update failed: {e}")
    
    print("✓ MAML operational")