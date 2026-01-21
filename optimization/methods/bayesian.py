"""Bayesian Optimization for SNN hyperparameters."""

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
from core.objective import ObjectiveFunction


@dataclass
class BOConfig:
    """Configuration for Bayesian Optimization."""
    n_iterations: int = 50
    n_initial: int = 10
    acquisition: str = 'ei'
    kappa: float = 2.0
    xi: float = 0.01
    seed: Optional[int] = None
    verbose: bool = True
    gp_length_scale: float = 0.5
    gp_noise: float = 0.1


@dataclass
class BOResult:
    """Result from Bayesian Optimization."""
    best_params: Dict[str, float]
    best_fitness: float
    best_spike_data: Optional[Dict] = None
    history: List[Dict] = field(default_factory=list)
    all_evaluations: List[Dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    n_evaluations: int = 0
    
    def to_dict(self) -> dict:
        return {'best_params': self.best_params, 'best_fitness': self.best_fitness,
                'history': self.history, 'all_evaluations': self.all_evaluations,
                'runtime_seconds': self.runtime_seconds, 'n_evaluations': self.n_evaluations}
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class GaussianProcessRegressor:
    """Simple GP regressor for Bayesian Optimization."""
    
    def __init__(self, length_scale: float = 0.5, noise: float = 0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        sq_dist = np.clip(sq_dist, 0, None)
        return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train, self.y_train = X.copy(), y.copy()
        K = self._rbf_kernel(X, X) + self.noise ** 2 * np.eye(len(X))
        try:
            self.K_inv = np.linalg.inv(K + 1e-8 * np.eye(len(K)))
            self.alpha = self.K_inv @ self.y_train
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ self.y_train
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))
        K_star = self._rbf_kernel(X, self.X_train)
        mean = K_star @ self.alpha
        K_star_star = self._rbf_kernel(X, X)
        var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        return mean, np.sqrt(np.clip(var, 1e-10, None))


class BayesianOptimizer:
    """Bayesian Optimization for SNN hyperparameter tuning."""
    
    def __init__(self, param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[BOConfig] = None,
                 network_config=None,
                 sim_config: Optional[SimulationConfig] = None,
                 conn_prob: float = 0.1):
        self.param_space = param_space or get_parameter_space()
        self.config = config or BOConfig()
        self.network_config = network_config
        self.sim_config = sim_config
        self.conn_prob = conn_prob
        self.objective_fn = objective_fn or ObjectiveFunction()
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.n_params = self.param_space.n_params
        self.gp = GaussianProcessRegressor(length_scale=self.config.gp_length_scale, noise=self.config.gp_noise)
        
        self.X_observed = []
        self.y_observed = []
        self.history = []
        self.n_evaluations = 0
        self._best_fitness = float('inf')
        self._best_params = None
        self._best_spike_data = None
    
    def _evaluate(self, params_dict: Dict[str, float], seed: int = None) -> float:
        self.n_evaluations += 1
        try:
            eval_seed = seed if seed is not None else (self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations)
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
    
    def _expected_improvement(self, X: np.ndarray, y_best: float) -> np.ndarray:
        mean, std = self.gp.predict(X)
        std = np.clip(std, 1e-10, None)
        z = (y_best - mean - self.config.xi) / std
        from scipy.stats import norm
        return (y_best - mean - self.config.xi) * norm.cdf(z) + std * norm.pdf(z)
    
    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        mean, std = self.gp.predict(X)
        return -(mean - self.config.kappa * std)
    
    def _acquisition(self, X: np.ndarray) -> np.ndarray:
        if len(self.y_observed) == 0:
            return np.random.rand(len(X))
        y_best = np.min(self.y_observed)
        if self.config.acquisition == 'ei':
            return self._expected_improvement(X, y_best)
        elif self.config.acquisition == 'ucb':
            return self._upper_confidence_bound(X)
        raise ValueError(f"Unknown acquisition: {self.config.acquisition}")
    
    def _optimize_acquisition(self, n_candidates: int = 1000) -> np.ndarray:
        candidates = np.random.rand(n_candidates, self.n_params)
        
        if len(self.X_observed) > 0:
            X_obs = np.array(self.X_observed)
            for x in X_obs[:min(10, len(X_obs))]:
                for _ in range(5):
                    perturbed = np.clip(x + np.random.randn(self.n_params) * 0.1, 0, 1)
                    candidates = np.vstack([candidates, perturbed])
        
        acq_values = self._acquisition(candidates)
        return candidates[np.argmax(acq_values)]
    
    def _initial_sampling(self) -> List[np.ndarray]:
        n, d = self.config.n_initial, self.n_params
        samples = np.zeros((n, d))
        for j in range(d):
            perm = np.random.permutation(n)
            samples[:, j] = (perm + np.random.rand(n)) / n
        samples[0] = self.param_space.normalize(self.param_space.get_default())
        return [samples[i] for i in range(n)]
    
    def optimize(self) -> BOResult:
        start_time = time.time()
        
        if self.config.verbose:
            print(f"Starting Bayesian Optimization\n  Iterations: {self.config.n_iterations}\n"
                  f"  Initial samples: {self.config.n_initial}\n  Acquisition: {self.config.acquisition}\n"
                  f"  Parameters: {self.n_params}\n")
        
        if self.config.verbose:
            print("Initial sampling phase...")
        
        initial_points = self._initial_sampling()
        all_evaluations = []
        
        for i, x in enumerate(initial_points):
            params = self.param_space.denormalize(x)
            fitness = self._evaluate(params)
            self.X_observed.append(x)
            self.y_observed.append(fitness)
            all_evaluations.append({'iteration': i, 'params': params, 'fitness': fitness, 'type': 'initial'})
            if self.config.verbose:
                print(f"  Initial {i+1}/{len(initial_points)}: fitness={fitness:.4f}")
        
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        
        best_idx = np.argmin(self.y_observed)
        best_fitness = self.y_observed[best_idx]
        best_x = self.X_observed[best_idx]
        
        self.history = [{'iteration': 0, 'best_fitness': float(best_fitness), 'current_fitness': float(best_fitness)}]
        
        if self.config.verbose:
            print(f"\nBO iterations...")
        
        n_bo_iters = self.config.n_iterations - self.config.n_initial
        
        for i in range(n_bo_iters):
            x_next = self._optimize_acquisition()
            params_next = self.param_space.denormalize(x_next)
            fitness = self._evaluate(params_next)
            
            self.X_observed.append(x_next)
            self.y_observed.append(fitness)
            all_evaluations.append({'iteration': self.config.n_initial + i, 'params': params_next,
                                    'fitness': fitness, 'type': 'bo'})
            
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_x = x_next.copy()
                if self.config.verbose:
                    print(f"  Iter {i+1}/{n_bo_iters}: NEW BEST fitness={fitness:.4f}")
            elif self.config.verbose and i % 5 == 0:
                print(f"  Iter {i+1}/{n_bo_iters}: fitness={fitness:.4f}, best={best_fitness:.4f}")
            
            self.history.append({'iteration': self.config.n_initial + i,
                                'best_fitness': float(best_fitness), 'current_fitness': float(fitness)})
        
        runtime = time.time() - start_time
        best_params = self._best_params if self._best_params else self.param_space.denormalize(best_x)
        
        if self.config.verbose:
            print(f"\nOptimization complete!\n  Best fitness: {self._best_fitness:.4f}\n"
                  f"  Runtime: {runtime:.1f}s\n  Evaluations: {self.n_evaluations}")
        
        return BOResult(best_params=best_params, best_fitness=float(self._best_fitness),
                       best_spike_data=self._best_spike_data, history=self.history,
                       all_evaluations=all_evaluations, runtime_seconds=runtime, n_evaluations=self.n_evaluations)


class OptunaOptimizer:
    """Wrapper for Optuna library if available."""
    
    def __init__(self, param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[BOConfig] = None,
                 network_config=None,
                 sim_config: Optional[SimulationConfig] = None,
                 conn_prob: float = 0.1):
        self.param_space = param_space or get_parameter_space()
        self.config = config or BOConfig()
        self.network_config = network_config
        self.sim_config = sim_config
        self.conn_prob = conn_prob
        self.objective_fn = objective_fn or ObjectiveFunction()
        self.n_evaluations = 0
        self.history = []
    
    def _objective(self, trial) -> float:
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Use BayesianOptimizer instead.")
        
        self.n_evaluations += 1
        params = {}
        for spec in self.param_space.specs:
            params[spec.name] = trial.suggest_float(spec.name, spec.min_val, spec.max_val, log=spec.log_scale)
        
        try:
            eval_seed = self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations
            spike_data = run_simulation(params, network_config=self.network_config,
                                        sim_config=self.sim_config, conn_prob=self.conn_prob, seed=eval_seed)
            return self.objective_fn(spike_data)
        except Exception:
            return 1e6
    
    def optimize(self) -> BOResult:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.INFO if self.config.verbose else optuna.logging.WARNING)
        except ImportError:
            print("Optuna not installed. Falling back to custom BO.")
            bo = BayesianOptimizer(self.param_space, self.objective_fn, self.config,
                                   network_config=self.network_config, sim_config=self.sim_config)
            return bo.optimize()
        
        start_time = time.time()
        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(self._objective, n_trials=self.config.n_iterations)
        
        runtime = time.time() - start_time
        
        history = []
        best_so_far = float('inf')
        for i, trial in enumerate(study.trials):
            if trial.value is not None and trial.value < best_so_far:
                best_so_far = trial.value
            history.append({'iteration': i, 'best_fitness': best_so_far, 'current_fitness': trial.value or 1e6})
        
        if self.config.verbose:
            print(f"\nOptimization complete!\n  Best fitness: {study.best_value:.4f}\n  Runtime: {runtime:.1f}s")
        
        return BOResult(best_params=study.best_params, best_fitness=float(study.best_value),
                       history=history, runtime_seconds=runtime, n_evaluations=self.n_evaluations)


def run_bayesian_optimization(param_space: Optional[ParameterSpace] = None,
                              n_iterations: int = 50, n_initial: int = 10,
                              seed: Optional[int] = None, use_optuna: bool = False,
                              network_config=None, sim_config: Optional[SimulationConfig] = None,
                              objective_fn: Optional[Callable] = None,
                              conn_prob: float = 0.1) -> BOResult:
    config = BOConfig(n_iterations=n_iterations, n_initial=n_initial, seed=seed)
    
    if use_optuna:
        optimizer = OptunaOptimizer(param_space, objective_fn=objective_fn, config=config,
                                    network_config=network_config, sim_config=sim_config, conn_prob=conn_prob)
    else:
        optimizer = BayesianOptimizer(param_space, objective_fn=objective_fn, config=config,
                                      network_config=network_config, sim_config=sim_config, conn_prob=conn_prob)
    return optimizer.optimize()


if __name__ == "__main__":
    from core.parameters import get_reduced_parameter_space
    print("Testing Bayesian Optimization with reduced parameter space...")
    result = run_bayesian_optimization(param_space=get_reduced_parameter_space(),
                                        n_iterations=20, n_initial=5, seed=42)
    print(f"\nBest parameters found:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value:.6f}")
