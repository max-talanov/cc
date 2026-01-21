"""Genetic Algorithm optimization for SNN hyperparameters."""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parameters import ParameterSpace, get_parameter_space
from core.simulator import run_simulation, SimulationConfig
from core.objective import ObjectiveFunction, MultiObjective


@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm."""
    population_size: int = 50
    n_generations: int = 30
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    mutation_eta: float = 20.0
    crossover_eta: float = 15.0
    tournament_size: int = 3
    elite_size: int = 2
    seed: Optional[int] = None
    verbose: bool = True
    multi_objective: bool = False
    n_reference_points: int = 12


@dataclass 
class GAResult:
    """Result from GA optimization."""
    best_params: Dict[str, float]
    best_fitness: float
    best_spike_data: Optional[Dict] = None
    history: List[Dict] = field(default_factory=list)
    pareto_front: Optional[List[Dict]] = None
    runtime_seconds: float = 0.0
    n_evaluations: int = 0
    
    def to_dict(self) -> dict:
        return {'best_params': self.best_params, 'best_fitness': self.best_fitness,
                'history': self.history, 'pareto_front': self.pareto_front,
                'runtime_seconds': self.runtime_seconds, 'n_evaluations': self.n_evaluations}
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class GeneticAlgorithm:
    """GA for SNN hyperparameter optimization."""
    
    def __init__(self, param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[GAConfig] = None,
                 network_config=None,
                 sim_config: Optional[SimulationConfig] = None,
                 conn_prob: float = 0.1):
        self.param_space = param_space or get_parameter_space()
        self.config = config or GAConfig()
        self.network_config = network_config
        self.sim_config = sim_config
        self.conn_prob = conn_prob
        self.objective_fn = objective_fn or ObjectiveFunction()
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.n_params = self.param_space.n_params
        self.history = []
        self.n_evaluations = 0
        self._best_fitness = float('inf')
        self._best_params = None
        self._best_spike_data = None
        
    def _evaluate(self, params_dict: Dict[str, float]) -> float:
        self.n_evaluations += 1
        try:
            eval_seed = self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations
            spike_data = run_simulation(params_dict, network_config=self.network_config,
                                        sim_config=self.sim_config, conn_prob=self.conn_prob, seed=eval_seed)
            fitness = self.objective_fn(spike_data)
            
            if fitness < self._best_fitness:
                self._best_fitness = fitness
                self._best_params = params_dict.copy()
                self._best_spike_data = {k: v.copy() for k, v in spike_data.items()}
            return fitness
        except Exception as e:
            if self.config.verbose:
                print(f"  Evaluation failed: {e}")
            return 1e6
    
    def _initialize_population(self) -> np.ndarray:
        population = np.random.rand(self.config.population_size, self.n_params)
        population[0] = self.param_space.normalize(self.param_space.get_default())
        return population
    
    def _decode_population(self, population: np.ndarray) -> List[Dict[str, float]]:
        return [self.param_space.denormalize(ind) for ind in population]
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.array([self._evaluate(p) for p in self._decode_population(population)])
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        n_selected = self.config.population_size - self.config.elite_size
        selected = []
        for _ in range(n_selected):
            idx = np.random.choice(len(population), self.config.tournament_size, replace=False)
            winner_idx = idx[np.argmin(fitness[idx])]
            selected.append(population[winner_idx].copy())
        return np.array(selected)
    
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.config.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = np.zeros(self.n_params), np.zeros(self.n_params)
        eta = self.config.crossover_eta
        
        for i in range(self.n_params):
            if np.random.rand() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    y1, y2 = (parent1[i], parent2[i]) if parent1[i] < parent2[i] else (parent2[i], parent1[i])
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * y1) / (y2 - y1 + 1e-10)
                    alpha = 2.0 - beta ** (-(eta + 1))
                    betaq = (rand * alpha) ** (1.0 / (eta + 1)) if rand <= 1.0 / alpha else (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                else:
                    child1[i], child2[i] = parent1[i], parent2[i]
            else:
                child1[i], child2[i] = parent1[i], parent2[i]
        
        return np.clip(child1, 0, 1), np.clip(child2, 0, 1)
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        eta = self.config.mutation_eta
        
        for i in range(self.n_params):
            if np.random.rand() < self.config.mutation_prob:
                y = mutated[i]
                delta1, delta2 = y, 1.0 - y
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
        elite_idx = np.argsort(fitness)[:self.config.elite_size]
        return population[elite_idx].copy()
    
    def optimize(self) -> GAResult:
        start_time = time.time()
        
        if self.config.verbose:
            print(f"Starting GA optimization\n  Population: {self.config.population_size}\n"
                  f"  Generations: {self.config.n_generations}\n  Parameters: {self.n_params}\n")
        
        population = self._initialize_population()
        fitness = self._evaluate_population(population)
        
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()
        
        self.history = [{'generation': 0, 'best_fitness': float(best_fitness),
                        'mean_fitness': float(np.mean(fitness)), 'std_fitness': float(np.std(fitness))}]
        
        if self.config.verbose:
            print(f"Gen 0: Best={best_fitness:.4f}, Mean={np.mean(fitness):.4f}")
        
        for gen in range(1, self.config.n_generations + 1):
            elite = self._get_elite(population, fitness)
            selected = self._tournament_selection(population, fitness)
            
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self._sbx_crossover(selected[i], selected[i+1])
                offspring.extend([self._polynomial_mutation(child1), self._polynomial_mutation(child2)])
            if len(offspring) < len(selected):
                offspring.append(self._polynomial_mutation(selected[-1]))
            
            new_population = np.vstack([elite, np.array(offspring[:len(selected)])])
            new_fitness = self._evaluate_population(new_population)
            
            gen_best_idx = np.argmin(new_fitness)
            if new_fitness[gen_best_idx] < best_fitness:
                best_fitness = new_fitness[gen_best_idx]
                best_individual = new_population[gen_best_idx].copy()
            
            population, fitness = new_population, new_fitness
            self.history.append({'generation': gen, 'best_fitness': float(best_fitness),
                                'mean_fitness': float(np.mean(fitness)), 'std_fitness': float(np.std(fitness))})
            
            if self.config.verbose and gen % 5 == 0:
                print(f"Gen {gen}: Best={best_fitness:.4f}, Mean={np.mean(fitness):.4f}")
        
        runtime = time.time() - start_time
        best_params = self._best_params if self._best_params else self.param_space.denormalize(best_individual)
        
        if self.config.verbose:
            print(f"\nOptimization complete!\n  Best fitness: {self._best_fitness:.4f}\n"
                  f"  Runtime: {runtime:.1f}s\n  Evaluations: {self.n_evaluations}")
        
        return GAResult(best_params=best_params, best_fitness=float(self._best_fitness),
                       best_spike_data=self._best_spike_data, history=self.history,
                       runtime_seconds=runtime, n_evaluations=self.n_evaluations)


class NSGA3:
    """NSGA-III for multi-objective optimization."""
    
    def __init__(self, param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[GAConfig] = None,
                 network_config=None,
                 sim_config: Optional[SimulationConfig] = None,
                 conn_prob: float = 0.1):
        self.param_space = param_space or get_parameter_space()
        self.config = config or GAConfig(multi_objective=True)
        self.network_config = network_config
        self.sim_config = sim_config
        self.conn_prob = conn_prob
        self.objective_fn = objective_fn or MultiObjective()
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.n_params = self.param_space.n_params
        self.n_objectives = self.objective_fn.n_objectives
        self.history = []
        self.n_evaluations = 0
        self.reference_points = np.random.dirichlet(np.ones(self.n_objectives), self.config.n_reference_points)
    
    def _evaluate(self, params_dict: Dict[str, float]) -> np.ndarray:
        self.n_evaluations += 1
        try:
            eval_seed = self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations
            spike_data = run_simulation(params_dict, network_config=self.network_config,
                                        sim_config=self.sim_config, conn_prob=self.conn_prob, seed=eval_seed)
            return self.objective_fn(spike_data)
        except Exception as e:
            if self.config.verbose:
                print(f"  Evaluation failed: {e}")
            return np.full(self.n_objectives, 1e6)
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
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
        return fronts[:-1]
    
    def _associate_to_reference_points(self, objectives: np.ndarray) -> np.ndarray:
        obj_min, obj_max = objectives.min(axis=0), objectives.max(axis=0)
        normalized = (objectives - obj_min) / (obj_max - obj_min + 1e-10)
        associations = np.zeros(len(objectives), dtype=int)
        for i, obj in enumerate(normalized):
            associations[i] = np.argmin(np.linalg.norm(self.reference_points - obj, axis=1))
        return associations
    
    def optimize(self) -> GAResult:
        start_time = time.time()
        
        if self.config.verbose:
            print(f"Starting NSGA-III multi-objective optimization\n  Population: {self.config.population_size}\n"
                  f"  Generations: {self.config.n_generations}\n  Objectives: {self.n_objectives}\n")
        
        population = np.random.rand(self.config.population_size, self.n_params)
        population[0] = self.param_space.normalize(self.param_space.get_default())
        
        params_list = [self.param_space.denormalize(ind) for ind in population]
        objectives = np.array([self._evaluate(p) for p in params_list])
        
        pareto_individuals, pareto_objectives = [], []
        ga = GeneticAlgorithm(self.param_space, config=GAConfig(crossover_prob=self.config.crossover_prob,
                                                                 mutation_prob=self.config.mutation_prob, verbose=False))
        
        for gen in range(self.config.n_generations):
            fronts = self._non_dominated_sort(objectives)
            
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= self.config.population_size:
                    selected_idx.extend(front)
                else:
                    remaining = self.config.population_size - len(selected_idx)
                    subset = np.random.choice(front, remaining, replace=False)
                    selected_idx.extend(subset)
                    break
            
            if fronts:
                pareto_individuals = [population[i].copy() for i in fronts[0]]
                pareto_objectives = [objectives[i].copy() for i in fronts[0]]
            
            offspring = []
            for _ in range(self.config.population_size):
                p1, p2 = np.random.choice(selected_idx, 2, replace=False)
                child1, _ = ga._sbx_crossover(population[p1], population[p2])
                offspring.append(ga._polynomial_mutation(child1))
            offspring = np.array(offspring)
            
            combined_pop = np.vstack([population[selected_idx], offspring])
            combined_params = [self.param_space.denormalize(ind) for ind in combined_pop]
            combined_obj = np.array([self._evaluate(p) for p in combined_params])
            
            fronts = self._non_dominated_sort(combined_obj)
            next_idx = []
            for front in fronts:
                if len(next_idx) + len(front) <= self.config.population_size:
                    next_idx.extend(front)
                else:
                    remaining = self.config.population_size - len(next_idx)
                    next_idx.extend(np.random.choice(front, remaining, replace=False))
                    break
            
            population, objectives = combined_pop[next_idx], combined_obj[next_idx]
            
            best_sum = np.min(np.sum(objectives, axis=1))
            self.history.append({'generation': gen, 'best_sum_objectives': float(best_sum),
                                'pareto_size': len(fronts[0]) if fronts else 0})
            
            if self.config.verbose and gen % 5 == 0:
                print(f"Gen {gen}: Pareto size={len(fronts[0]) if fronts else 0}, Best sum={best_sum:.4f}")
        
        runtime = time.time() - start_time
        
        obj_sums = np.sum(objectives, axis=1)
        best_idx = np.argmin(obj_sums)
        best_params = self.param_space.denormalize(population[best_idx])
        best_fitness = float(obj_sums[best_idx])
        
        pareto_front = [{'params': self.param_space.denormalize(ind), 'objectives': obj.tolist()}
                        for ind, obj in zip(pareto_individuals, pareto_objectives)]
        
        if self.config.verbose:
            print(f"\nOptimization complete!\n  Pareto front size: {len(pareto_front)}\n"
                  f"  Best compromise fitness: {best_fitness:.4f}\n  Runtime: {runtime:.1f}s")
        
        return GAResult(best_params=best_params, best_fitness=best_fitness, history=self.history,
                       pareto_front=pareto_front, runtime_seconds=runtime, n_evaluations=self.n_evaluations)


def run_ga(param_space: Optional[ParameterSpace] = None, n_generations: int = 30,
           population_size: int = 50, seed: Optional[int] = None, network_config=None,
           sim_config: Optional[SimulationConfig] = None, objective_fn: Optional[Callable] = None,
           conn_prob: float = 0.1) -> GAResult:
    config = GAConfig(n_generations=n_generations, population_size=population_size, seed=seed)
    ga = GeneticAlgorithm(param_space, objective_fn=objective_fn, config=config,
                          network_config=network_config, sim_config=sim_config, conn_prob=conn_prob)
    return ga.optimize()


def run_nsga3(param_space: Optional[ParameterSpace] = None, n_generations: int = 30,
              population_size: int = 50, seed: Optional[int] = None, network_config=None,
              sim_config: Optional[SimulationConfig] = None, objective_fn: Optional[Callable] = None,
              conn_prob: float = 0.1) -> GAResult:
    config = GAConfig(n_generations=n_generations, population_size=population_size, seed=seed, multi_objective=True)
    nsga = NSGA3(param_space, objective_fn=objective_fn, config=config,
                 network_config=network_config, sim_config=sim_config, conn_prob=conn_prob)
    return nsga.optimize()


if __name__ == "__main__":
    from core.parameters import get_reduced_parameter_space
    print("Testing GA with reduced parameter space...")
    result = run_ga(param_space=get_reduced_parameter_space(), n_generations=10, population_size=20, seed=42)
    print(f"\nBest parameters found:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value:.6f}")
