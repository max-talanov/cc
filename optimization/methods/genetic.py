"""
Genetic Algorithm optimization for SNN hyperparameters.
Supports both single-objective (GA) and multi-objective (NSGA-III).
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parameters import ParameterSpace, get_parameter_space
from core.simulator import run_simulation
from core.objective import ObjectiveFunction, MultiObjective


@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm."""
    population_size: int = 50
    n_generations: int = 30
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    mutation_eta: float = 20.0  # Distribution index for polynomial mutation
    crossover_eta: float = 15.0  # Distribution index for SBX crossover
    tournament_size: int = 3
    elite_size: int = 2  # Number of best individuals to preserve
    seed: Optional[int] = None
    verbose: bool = True
    
    # Multi-objective specific
    multi_objective: bool = False
    n_reference_points: int = 12  # For NSGA-III


@dataclass 
class GAResult:
    """Result from GA optimization."""
    best_params: Dict[str, float]
    best_fitness: float
    best_spike_data: Optional[Dict] = None  # Store the actual spike data
    history: List[Dict] = field(default_factory=list)
    pareto_front: Optional[List[Dict]] = None  # For multi-objective
    runtime_seconds: float = 0.0
    n_evaluations: int = 0
    
    def to_dict(self) -> dict:
        return {
            'best_params': self.best_params,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'pareto_front': self.pareto_front,
            'runtime_seconds': self.runtime_seconds,
            'n_evaluations': self.n_evaluations,
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class GeneticAlgorithm:
    """
    Genetic Algorithm for SNN hyperparameter optimization.
    
    Based on methodology from:
    Fitzgerald & Wong-Lin (2021) - Multi-Objective Optimisation of Cortical 
    Spiking Neural Networks With Genetic Algorithms
    """
    
    def __init__(self, 
                 param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[GAConfig] = None,
                 network_config = None):
        
        self.param_space = param_space or get_parameter_space()
        self.config = config or GAConfig()
        self.network_config = network_config  # Network architecture config
        
        if objective_fn is None:
            self.objective_fn = ObjectiveFunction()
        else:
            self.objective_fn = objective_fn
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.n_params = self.param_space.n_params
        self.history = []
        self.n_evaluations = 0
        
        # Track best result with spike data
        self._best_fitness = float('inf')
        self._best_params = None
        self._best_spike_data = None
        
    def _evaluate(self, params_dict: Dict[str, float]) -> float:
        """Evaluate a single parameter set."""
        self.n_evaluations += 1
        try:
            # Use consistent seed for reproducibility
            eval_seed = self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations
            spike_data = run_simulation(params_dict, network_config=self.network_config, seed=eval_seed)
            fitness = self.objective_fn(spike_data)
            
            # Store if this is the best result
            if fitness < self._best_fitness:
                self._best_fitness = fitness
                self._best_params = params_dict.copy()
                self._best_spike_data = {k: v.copy() for k, v in spike_data.items()}
            
            return fitness
            return fitness
        except Exception as e:
            if self.config.verbose:
                print(f"  Evaluation failed: {e}")
            return 1e6  # Penalty for failed evaluation
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population in normalized [0, 1] space."""
        pop_size = self.config.population_size
        
        # Random initialization
        population = np.random.rand(pop_size, self.n_params)
        
        # Include default parameters as one individual
        default_params = self.param_space.get_default()
        population[0] = self.param_space.normalize(default_params)
        
        return population
    
    def _decode_population(self, population: np.ndarray) -> List[Dict[str, float]]:
        """Convert normalized population to parameter dicts."""
        return [self.param_space.denormalize(ind) for ind in population]
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of entire population."""
        params_list = self._decode_population(population)
        fitness = np.array([self._evaluate(p) for p in params_list])
        return fitness
    
    def _tournament_selection(self, population: np.ndarray, 
                             fitness: np.ndarray) -> np.ndarray:
        """Select parents via tournament selection."""
        n_selected = self.config.population_size - self.config.elite_size
        selected = []
        
        for _ in range(n_selected):
            # Random tournament
            idx = np.random.choice(len(population), self.config.tournament_size, replace=False)
            tournament_fitness = fitness[idx]
            winner_idx = idx[np.argmin(tournament_fitness)]  # Lower is better
            selected.append(population[winner_idx].copy())
        
        return np.array(selected)
    
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        if np.random.rand() > self.config.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        child1 = np.zeros(self.n_params)
        child2 = np.zeros(self.n_params)
        eta = self.config.crossover_eta
        
        for i in range(self.n_params):
            if np.random.rand() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    # SBX formula
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * y1) / (y2 - y1 + 1e-10)
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    
                    child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        # Clip to bounds
        child1 = np.clip(child1, 0, 1)
        child2 = np.clip(child2, 0, 1)
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        mutated = individual.copy()
        eta = self.config.mutation_eta
        
        for i in range(self.n_params):
            if np.random.rand() < self.config.mutation_prob:
                y = mutated[i]
                delta1 = y
                delta2 = 1.0 - y
                
                rand = np.random.rand()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    deltaq = 1.0 - val ** mut_pow
                
                mutated[i] = np.clip(y + deltaq, 0, 1)
        
        return mutated
    
    def _get_elite(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Get elite individuals (best performers)."""
        elite_idx = np.argsort(fitness)[:self.config.elite_size]
        return population[elite_idx].copy()
    
    def optimize(self) -> GAResult:
        """Run the genetic algorithm optimization."""
        start_time = time.time()
        
        if self.config.verbose:
            print(f"Starting GA optimization")
            print(f"  Population: {self.config.population_size}")
            print(f"  Generations: {self.config.n_generations}")
            print(f"  Parameters: {self.n_params}")
            print()
        
        # Initialize
        population = self._initialize_population()
        fitness = self._evaluate_population(population)
        
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()
        
        self.history = [{
            'generation': 0,
            'best_fitness': float(best_fitness),
            'mean_fitness': float(np.mean(fitness)),
            'std_fitness': float(np.std(fitness)),
        }]
        
        if self.config.verbose:
            print(f"Gen 0: Best={best_fitness:.4f}, Mean={np.mean(fitness):.4f}")
        
        # Evolution loop
        for gen in range(1, self.config.n_generations + 1):
            # Selection
            elite = self._get_elite(population, fitness)
            selected = self._tournament_selection(population, fitness)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self._sbx_crossover(selected[i], selected[i+1])
                child1 = self._polynomial_mutation(child1)
                child2 = self._polynomial_mutation(child2)
                offspring.extend([child1, child2])
            
            # Handle odd number
            if len(offspring) < len(selected):
                offspring.append(self._polynomial_mutation(selected[-1]))
            
            # New population = elite + offspring
            new_population = np.vstack([elite, np.array(offspring[:len(selected)])])
            
            # Evaluate new population
            new_fitness = self._evaluate_population(new_population)
            
            # Update best
            gen_best_idx = np.argmin(new_fitness)
            if new_fitness[gen_best_idx] < best_fitness:
                best_fitness = new_fitness[gen_best_idx]
                best_individual = new_population[gen_best_idx].copy()
            
            population = new_population
            fitness = new_fitness
            
            # Record history
            self.history.append({
                'generation': gen,
                'best_fitness': float(best_fitness),
                'mean_fitness': float(np.mean(fitness)),
                'std_fitness': float(np.std(fitness)),
            })
            
            if self.config.verbose and gen % 5 == 0:
                print(f"Gen {gen}: Best={best_fitness:.4f}, Mean={np.mean(fitness):.4f}")
        
        runtime = time.time() - start_time
        
        # Use stored best params for consistency
        best_params = self._best_params if self._best_params else self.param_space.denormalize(best_individual)
        best_fitness = self._best_fitness
        
        if self.config.verbose:
            print(f"\nOptimization complete!")
            print(f"  Best fitness: {best_fitness:.4f}")
            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Evaluations: {self.n_evaluations}")
        
        return GAResult(
            best_params=best_params,
            best_fitness=float(best_fitness),
            best_spike_data=self._best_spike_data,
            history=self.history,
            runtime_seconds=runtime,
            n_evaluations=self.n_evaluations,
        )


class NSGA3:
    """
    NSGA-III for multi-objective optimization.
    Simplified implementation based on Deb & Jain (2014).
    """
    
    def __init__(self,
                 param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[GAConfig] = None,
                 network_config = None):
        
        self.param_space = param_space or get_parameter_space()
        self.config = config or GAConfig(multi_objective=True)
        self.network_config = network_config  # Network architecture config
        
        if objective_fn is None:
            self.objective_fn = MultiObjective()
        else:
            self.objective_fn = objective_fn
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.n_params = self.param_space.n_params
        self.n_objectives = self.objective_fn.n_objectives
        self.history = []
        self.n_evaluations = 0
        
        # Generate reference points
        self.reference_points = self._generate_reference_points()
    
    def _generate_reference_points(self) -> np.ndarray:
        """Generate Das-Dennis reference points for NSGA-III."""
        # Simplified: uniform grid on simplex
        n_points = self.config.n_reference_points
        n_obj = self.n_objectives
        
        # For simplicity, use random points on simplex
        points = np.random.dirichlet(np.ones(n_obj), n_points)
        return points
    
    def _evaluate(self, params_dict: Dict[str, float]) -> np.ndarray:
        """Evaluate and return objective vector."""
        self.n_evaluations += 1
        try:
            eval_seed = self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations
            spike_data = run_simulation(params_dict, network_config=self.network_config, seed=eval_seed)
            objectives = self.objective_fn(spike_data)
            return objectives
        except Exception as e:
            if self.config.verbose:
                print(f"  Evaluation failed: {e}")
            return np.full(self.n_objectives, 1e6)
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (all <= and at least one <)."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting."""
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _associate_to_reference_points(self, objectives: np.ndarray) -> np.ndarray:
        """Associate each solution to nearest reference point."""
        # Normalize objectives
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min + 1e-10
        normalized = (objectives - obj_min) / obj_range
        
        # Find nearest reference point for each solution
        associations = np.zeros(len(objectives), dtype=int)
        for i, obj in enumerate(normalized):
            distances = np.linalg.norm(self.reference_points - obj, axis=1)
            associations[i] = np.argmin(distances)
        
        return associations
    
    def optimize(self) -> GAResult:
        """Run NSGA-III optimization."""
        start_time = time.time()
        
        if self.config.verbose:
            print(f"Starting NSGA-III multi-objective optimization")
            print(f"  Population: {self.config.population_size}")
            print(f"  Generations: {self.config.n_generations}")
            print(f"  Objectives: {self.n_objectives}")
            print()
        
        # Initialize population
        population = np.random.rand(self.config.population_size, self.n_params)
        
        # Include default
        default_params = self.param_space.get_default()
        population[0] = self.param_space.normalize(default_params)
        
        # Evaluate
        params_list = [self.param_space.denormalize(ind) for ind in population]
        objectives = np.array([self._evaluate(p) for p in params_list])
        
        # Track Pareto front
        pareto_individuals = []
        pareto_objectives = []
        
        ga = GeneticAlgorithm(self.param_space, config=GAConfig(
            crossover_prob=self.config.crossover_prob,
            mutation_prob=self.config.mutation_prob,
            verbose=False
        ))
        
        for gen in range(self.config.n_generations):
            # Non-dominated sorting
            fronts = self._non_dominated_sort(objectives)
            
            # Select for next generation based on fronts
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= self.config.population_size:
                    selected_idx.extend(front)
                else:
                    # Need to select subset from this front
                    remaining = self.config.population_size - len(selected_idx)
                    # Use reference point association for niching
                    associations = self._associate_to_reference_points(objectives[front])
                    # Randomly select to fill
                    subset = np.random.choice(front, remaining, replace=False)
                    selected_idx.extend(subset)
                    break
            
            # Store first front as Pareto approximation
            if fronts:
                pareto_individuals = [population[i].copy() for i in fronts[0]]
                pareto_objectives = [objectives[i].copy() for i in fronts[0]]
            
            # Create offspring via crossover and mutation
            offspring = []
            for _ in range(self.config.population_size):
                p1, p2 = np.random.choice(selected_idx, 2, replace=False)
                child1, _ = ga._sbx_crossover(population[p1], population[p2])
                child1 = ga._polynomial_mutation(child1)
                offspring.append(child1)
            
            offspring = np.array(offspring)
            
            # Combine parent and offspring
            combined_pop = np.vstack([population[selected_idx], offspring])
            combined_params = [self.param_space.denormalize(ind) for ind in combined_pop]
            combined_obj = np.array([self._evaluate(p) for p in combined_params])
            
            # Select next generation
            fronts = self._non_dominated_sort(combined_obj)
            next_idx = []
            for front in fronts:
                if len(next_idx) + len(front) <= self.config.population_size:
                    next_idx.extend(front)
                else:
                    remaining = self.config.population_size - len(next_idx)
                    subset = np.random.choice(front, remaining, replace=False)
                    next_idx.extend(subset)
                    break
            
            population = combined_pop[next_idx]
            objectives = combined_obj[next_idx]
            
            # Record history
            best_sum = np.min(np.sum(objectives, axis=1))
            self.history.append({
                'generation': gen,
                'best_sum_objectives': float(best_sum),
                'pareto_size': len(fronts[0]) if fronts else 0,
            })
            
            if self.config.verbose and gen % 5 == 0:
                print(f"Gen {gen}: Pareto size={len(fronts[0]) if fronts else 0}, "
                      f"Best sum={best_sum:.4f}")
        
        runtime = time.time() - start_time
        
        # Find best compromise solution (minimum sum of objectives)
        obj_sums = np.sum(objectives, axis=1)
        best_idx = np.argmin(obj_sums)
        best_params = self.param_space.denormalize(population[best_idx])
        best_fitness = float(obj_sums[best_idx])
        
        # Build Pareto front result
        pareto_front = []
        for ind, obj in zip(pareto_individuals, pareto_objectives):
            pareto_front.append({
                'params': self.param_space.denormalize(ind),
                'objectives': obj.tolist(),
            })
        
        if self.config.verbose:
            print(f"\nOptimization complete!")
            print(f"  Pareto front size: {len(pareto_front)}")
            print(f"  Best compromise fitness: {best_fitness:.4f}")
            print(f"  Runtime: {runtime:.1f}s")
        
        return GAResult(
            best_params=best_params,
            best_fitness=best_fitness,
            history=self.history,
            pareto_front=pareto_front,
            runtime_seconds=runtime,
            n_evaluations=self.n_evaluations,
        )


# Convenience functions
def run_ga(param_space: Optional[ParameterSpace] = None,
           n_generations: int = 30,
           population_size: int = 50,
           seed: Optional[int] = None,
           network_config = None) -> GAResult:
    """Run single-objective GA optimization."""
    config = GAConfig(
        n_generations=n_generations,
        population_size=population_size,
        seed=seed,
    )
    ga = GeneticAlgorithm(param_space, config=config, network_config=network_config)
    return ga.optimize()


def run_nsga3(param_space: Optional[ParameterSpace] = None,
              n_generations: int = 30,
              population_size: int = 50,
              seed: Optional[int] = None,
              network_config = None) -> GAResult:
    """Run multi-objective NSGA-III optimization."""
    config = GAConfig(
        n_generations=n_generations,
        population_size=population_size,
        seed=seed,
        multi_objective=True,
    )
    nsga = NSGA3(param_space, config=config, network_config=network_config)
    return nsga.optimize()


if __name__ == "__main__":
    # Quick test
    from core.parameters import get_reduced_parameter_space
    
    print("Testing GA with reduced parameter space...")
    result = run_ga(
        param_space=get_reduced_parameter_space(),
        n_generations=10,
        population_size=20,
        seed=42
    )
    
    print(f"\nBest parameters found:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value:.6f}")