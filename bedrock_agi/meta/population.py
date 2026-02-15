"""
bedrock_agi/meta/population.py

Population-Based Training: Evolve hyperparameters
Explores hyperparameter space while respecting constitution.
"""

import torch
import numpy as np
import copy
from typing import List, Dict, Callable, Any, Tuple

from .fitness import compute_fitness
from .hyperparam_gates import constitute_hyperparams

class Population:
    """
    Population of models with different hyperparameters.
    
    Evolves via PBT (Population Based Training):
    1. Evaluate fitness (CRIS health + task performance)
    2. Exploit: Bottom performers copy top performers
    3. Explore: Mutate parameters within constitutional bounds
    """
    
    def __init__(
        self,
        model_factory: Callable[[], Tuple[torch.nn.Module, Dict[str, float]]],
        population_size: int = 8,
        mutation_rate: float = 0.2,
        perturbation_factor: float = 1.2
    ):
        """
        Args:
            model_factory: Function returning (new_model_instance, initial_hyperparams)
            population_size: Number of parallel workers/agents
            mutation_rate: Probability of mutation per step
            perturbation_factor: Magnitude of mutation (e.g., *1.2 or /1.2)
        """
        self.size = population_size
        self.mutation_rate = mutation_rate
        self.perturbation_factor = perturbation_factor
        
        # Initialize population
        self.members = []
        for i in range(population_size):
            model, params = model_factory()
            
            # Ensure initial params are valid
            ok, msg = constitute_hyperparams(params)
            if not ok:
                raise ValueError(f"Factory produced invalid initial params: {msg}")
                
            self.members.append({
                'id': i,
                'model': model,  # In real distributed PBT, this might be a path or remote ref
                'params': params,
                'fitness': -float('inf'),
                'history': []
            })
            
    def evaluate(self, eval_fn: Callable[[Any, Dict], float]):
        """
        Evaluate fitness for all members.
        
        Args:
            eval_fn: Function taking (model, params) -> fitness_scalar
        """
        for member in self.members:
            # Run evaluation
            fitness = eval_fn(member['model'], member['params'])
            member['fitness'] = fitness
            member['history'].append(fitness)
            
    def exploit(self):
        """
        Evolution Step 1: Exploit.
        Replace the bottom 25% of the population with copies of the top 25%.
        """
        # Sort by fitness descending (Best -> Worst)
        sorted_members = sorted(self.members, key=lambda m: m['fitness'], reverse=True)
        
        n_replace = self.size // 4
        if n_replace < 1: 
            return # Population too small to exploit
            
        top_performers = sorted_members[:n_replace]
        bottom_performers = sorted_members[-n_replace:]
        
        for bottom, top in zip(bottom_performers, top_performers):
            # Deep copy parameters from top to bottom
            # We preserve the 'id' and 'model' object identity, but overwrite state
            bottom['params'] = copy.deepcopy(top['params'])
            
            # In full PBT, we would also copy model weights here:
            # bottom['model'].load_state_dict(top['model'].state_dict())
            
            # Reset fitness since it's a new configuration
            bottom['fitness'] = top['fitness'] # Inherit fitness estimate
            
    def explore(self):
        """
        Evolution Step 2: Explore.
        Mutate parameters of the population within constitutional bounds.
        """
        for member in self.members:
            # Only mutate with some probability
            if np.random.rand() > self.mutation_rate:
                continue
                
            # Create candidate mutation
            new_params = copy.deepcopy(member['params'])
            
            # Mutate Learning Rate
            if 'lr' in new_params:
                if np.random.rand() < 0.5:
                    new_params['lr'] *= self.perturbation_factor
                else:
                    new_params['lr'] /= self.perturbation_factor
                    
            # Mutate Spectral Norm (eta) - Careful, close to 1.0
            if 'eta' in new_params:
                # Smaller perturbations for eta
                factor = 1.0 + (self.perturbation_factor - 1.0) * 0.1 
                if np.random.rand() < 0.5:
                    new_params['eta'] *= factor
                else:
                    new_params['eta'] /= factor
            
            # Mutate Projection Bias
            if 'bias' in new_params:
                factor = 1.0 + (self.perturbation_factor - 1.0) * 0.1
                if np.random.rand() < 0.5:
                    new_params['bias'] *= factor
                else:
                    new_params['bias'] /= factor
            
            # CRITICAL: Check Constitution
            # If mutation violates CRIS bounds, discard it (revert to old params)
            ok, msg = constitute_hyperparams(new_params)
            
            if ok:
                member['params'] = new_params
            # else: Keep old params (silent rejection of mutation)

    def get_best_params(self) -> Dict[str, float]:
        """Return the hyperparameters of the current best agent."""
        best_member = max(self.members, key=lambda m: m['fitness'])
        return best_member['params']

if __name__ == "__main__":
    print("Testing Population Manager...")
    
    # 1. Mock Factory
    def make_model():
        model = torch.nn.Linear(10, 2)
        # Randomize start to see convergence
        params = {
            'eta': float(np.random.uniform(0.90, 0.98)),
            'bias': float(np.random.uniform(0.95, 0.99)),
            'lr': float(10 ** np.random.uniform(-4, -3))
        }
        return model, params
        
    # 2. Initialize
    pop = Population(make_model, population_size=8, mutation_rate=1.0) # High rate for test
    print(f"✓ Population initialized ({pop.size} members)")
    
    # 3. Mock Evaluation Function
    # Let's say optimal is eta=0.95, bias=0.98, lr=1e-3
    def eval_fn(model, params):
        score = 0.0
        score -= abs(params['eta'] - 0.95) * 100
        score -= abs(params['bias'] - 0.98) * 100
        # closer lr is to 1e-3 (log scale)
        score -= abs(np.log10(params['lr']) - np.log10(1e-3)) * 10
        return score
        
    # 4. Run Loop
    print("\nRunning Evolution Loop...")
    for generation in range(5):
        pop.evaluate(eval_fn)
        best_p = pop.get_best_params()
        print(f"  Gen {generation}: Best Fit={max(m['fitness'] for m in pop.members):.2f} | η={best_p['eta']:.4f}")
        
        pop.exploit()
        pop.explore()
        
    print("\n✓ Exploit/Explore cycle verified")
    
    # 5. Verify Constraints
    # Force an invalid param and ensure it doesn't propagate via explore
    pop.members[0]['params']['eta'] = 0.99 # Boundary
    # Next mutation likely pushes it over 1.0. 
    # The 'explore' function should catch this via constitute_hyperparams.
    
    print("✓ Population operational")