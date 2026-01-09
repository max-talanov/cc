"""
Bayesian Optimization for SNN hyperparameters.
Uses Gaussian Process surrogate with Expected Improvement acquisition.
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
from core.objective import ObjectiveFunction


@dataclass
class BOConfig:
    """Configuration for Bayesian Optimization."""
    n_iterations: int = 50
    n_initial: int = 10  # Initial random samples before BO
    acquisition: str = 'ei'  # 'ei' (Expected Improvement) or 'ucb'
    kappa: float = 2.0  # Exploration parameter for UCB
    xi: float = 0.01  # Exploration parameter for EI
    seed: Optional[int] = None
    verbose: bool = True
    
    # GP hyperparameters
    gp_length_scale: float = 0.5
    gp_noise: float = 0.1


@dataclass
class BOResult:
    """Result from Bayesian Optimization."""
    best_params: Dict[str, float]
    best_fitness: float
    best_spike_data: Optional[Dict] = None  # Store the actual spike data
    history: List[Dict] = field(default_factory=list)
    all_evaluations: List[Dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    n_evaluations: int = 0
    
    def to_dict(self) -> dict:
        return {
            'best_params': self.best_params,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'all_evaluations': self.all_evaluations,
            'runtime_seconds': self.runtime_seconds,
            'n_evaluations': self.n_evaluations,
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class GaussianProcessRegressor:
    """
    Simple Gaussian Process regressor for Bayesian Optimization.
    Uses RBF kernel with automatic relevance determination.
    """
    
    def __init__(self, length_scale: float = 0.5, noise: float = 0.1):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        # Compute squared distances
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
                  np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        sq_dist = np.clip(sq_dist, 0, None)  # Numerical stability
        return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Compute kernel matrix
        K = self._rbf_kernel(X, X)
        K += self.noise ** 2 * np.eye(len(X))  # Add noise
        
        # Stable inversion
        try:
            self.K_inv = np.linalg.inv(K + 1e-8 * np.eye(len(K)))
            self.alpha = self.K_inv @ self.y_train
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.K_inv = np.linalg.pinv(K)
            self.alpha = self.K_inv @ self.y_train
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and std for new points."""
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))
        
        # Kernel between test and train
        K_star = self._rbf_kernel(X, self.X_train)
        
        # Mean prediction
        mean = K_star @ self.alpha
        
        # Variance prediction
        K_star_star = self._rbf_kernel(X, X)
        var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        var = np.clip(var, 1e-10, None)  # Ensure positive
        std = np.sqrt(var)
        
        return mean, std


class BayesianOptimizer:
    """
    Bayesian Optimization for SNN hyperparameter tuning.
    
    Uses a Gaussian Process surrogate model to guide the search
    towards promising regions of the parameter space.
    """
    
    def __init__(self,
                 param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[BOConfig] = None,
                 network_config = None):
        
        self.param_space = param_space or get_parameter_space()
        self.config = config or BOConfig()
        self.network_config = network_config  # Network architecture config
        
        if objective_fn is None:
            self.objective_fn = ObjectiveFunction()
        else:
            self.objective_fn = objective_fn
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.n_params = self.param_space.n_params
        self.gp = GaussianProcessRegressor(
            length_scale=self.config.gp_length_scale,
            noise=self.config.gp_noise
        )
        
        self.X_observed = []
        self.y_observed = []
        self.history = []
        self.n_evaluations = 0
        
        # Track best result with spike data
        self._best_fitness = float('inf')
        self._best_params = None
        self._best_spike_data = None
    
    def _evaluate(self, params_dict: Dict[str, float], seed: int = None) -> float:
        """Evaluate a single parameter set."""
        self.n_evaluations += 1
        try:
            # Use consistent seed for reproducibility
            eval_seed = seed if seed is not None else (self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations)
            spike_data = run_simulation(params_dict, network_config=self.network_config, seed=eval_seed)
            fitness = self.objective_fn(spike_data)
            
            # Store if this is the best result
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
        """Compute Expected Improvement acquisition function."""
        mean, std = self.gp.predict(X)
        
        # Handle zero std
        std = np.clip(std, 1e-10, None)
        
        # Z-score
        z = (y_best - mean - self.config.xi) / std
        
        # EI formula (minimization)
        from scipy.stats import norm
        ei = (y_best - mean - self.config.xi) * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
    
    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """Compute Upper Confidence Bound (for minimization, actually LCB)."""
        mean, std = self.gp.predict(X)
        
        # Lower confidence bound for minimization
        lcb = mean - self.config.kappa * std
        
        # Return negative for maximization of acquisition
        return -lcb
    
    def _acquisition(self, X: np.ndarray) -> np.ndarray:
        """Compute acquisition function values."""
        if len(self.y_observed) == 0:
            return np.random.rand(len(X))
        
        y_best = np.min(self.y_observed)
        
        if self.config.acquisition == 'ei':
            return self._expected_improvement(X, y_best)
        elif self.config.acquisition == 'ucb':
            return self._upper_confidence_bound(X)
        else:
            raise ValueError(f"Unknown acquisition: {self.config.acquisition}")
    
    def _optimize_acquisition(self, n_candidates: int = 1000) -> np.ndarray:
        """Find point that maximizes acquisition function."""
        # Random candidates
        candidates = np.random.rand(n_candidates, self.n_params)
        
        # Also include promising mutations of observed points
        if len(self.X_observed) > 0:
            X_obs = np.array(self.X_observed)
            for x in X_obs[:min(10, len(X_obs))]:
                # Small perturbations
                for _ in range(5):
                    perturbed = x + np.random.randn(self.n_params) * 0.1
                    perturbed = np.clip(perturbed, 0, 1)
                    candidates = np.vstack([candidates, perturbed])
        
        # Evaluate acquisition
        acq_values = self._acquisition(candidates)
        
        # Return best
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]
    
    def _initial_sampling(self) -> List[np.ndarray]:
        """Generate initial samples using Latin Hypercube Sampling."""
        n = self.config.n_initial
        d = self.n_params
        
        # LHS: divide each dimension into n intervals
        samples = np.zeros((n, d))
        for j in range(d):
            perm = np.random.permutation(n)
            samples[:, j] = (perm + np.random.rand(n)) / n
        
        # Include default parameters
        default_params = self.param_space.get_default()
        default_normalized = self.param_space.normalize(default_params)
        samples[0] = default_normalized
        
        return [samples[i] for i in range(n)]
    
    def optimize(self) -> BOResult:
        """Run Bayesian Optimization."""
        start_time = time.time()
        
        if self.config.verbose:
            print(f"Starting Bayesian Optimization")
            print(f"  Iterations: {self.config.n_iterations}")
            print(f"  Initial samples: {self.config.n_initial}")
            print(f"  Acquisition: {self.config.acquisition}")
            print(f"  Parameters: {self.n_params}")
            print()
        
        # Initial sampling
        if self.config.verbose:
            print("Initial sampling phase...")
        
        initial_points = self._initial_sampling()
        all_evaluations = []
        
        for i, x in enumerate(initial_points):
            params = self.param_space.denormalize(x)
            fitness = self._evaluate(params)
            
            self.X_observed.append(x)
            self.y_observed.append(fitness)
            
            all_evaluations.append({
                'iteration': i,
                'params': params,
                'fitness': fitness,
                'type': 'initial',
            })
            
            if self.config.verbose:
                print(f"  Initial {i+1}/{len(initial_points)}: fitness={fitness:.4f}")
        
        # Fit GP on initial data
        self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
        
        best_idx = np.argmin(self.y_observed)
        best_fitness = self.y_observed[best_idx]
        best_x = self.X_observed[best_idx]
        
        self.history = [{
            'iteration': 0,
            'best_fitness': float(best_fitness),
            'current_fitness': float(best_fitness),
        }]
        
        if self.config.verbose:
            print(f"\nBO iterations...")
        
        # BO iterations
        n_bo_iters = self.config.n_iterations - self.config.n_initial
        
        for i in range(n_bo_iters):
            # Find next point
            x_next = self._optimize_acquisition()
            params_next = self.param_space.denormalize(x_next)
            
            # Evaluate
            fitness = self._evaluate(params_next)
            
            # Update
            self.X_observed.append(x_next)
            self.y_observed.append(fitness)
            
            all_evaluations.append({
                'iteration': self.config.n_initial + i,
                'params': params_next,
                'fitness': fitness,
                'type': 'bo',
            })
            
            # Refit GP
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            
            # Track best
            if fitness < best_fitness:
                best_fitness = fitness
                best_x = x_next.copy()
                if self.config.verbose:
                    print(f"  Iter {i+1}/{n_bo_iters}: NEW BEST fitness={fitness:.4f}")
            elif self.config.verbose and i % 5 == 0:
                print(f"  Iter {i+1}/{n_bo_iters}: fitness={fitness:.4f}, best={best_fitness:.4f}")
            
            self.history.append({
                'iteration': self.config.n_initial + i,
                'best_fitness': float(best_fitness),
                'current_fitness': float(fitness),
            })
        
        runtime = time.time() - start_time
        
        # Use stored best params and spike data for consistency
        best_params = self._best_params if self._best_params else self.param_space.denormalize(best_x)
        best_fitness = self._best_fitness
        
        if self.config.verbose:
            print(f"\nOptimization complete!")
            print(f"  Best fitness: {best_fitness:.4f}")
            print(f"  Runtime: {runtime:.1f}s")
            print(f"  Evaluations: {self.n_evaluations}")
        
        return BOResult(
            best_params=best_params,
            best_fitness=float(best_fitness),
            best_spike_data=self._best_spike_data,
            history=self.history,
            all_evaluations=all_evaluations,
            runtime_seconds=runtime,
            n_evaluations=self.n_evaluations,
        )


class OptunaOptimizer:
    """
    Wrapper for Optuna library if available.
    Optuna provides more sophisticated BO with TPE sampler.
    """
    
    def __init__(self,
                 param_space: Optional[ParameterSpace] = None,
                 objective_fn: Optional[Callable] = None,
                 config: Optional[BOConfig] = None,
                 network_config = None):
        
        self.param_space = param_space or get_parameter_space()
        self.config = config or BOConfig()
        self.network_config = network_config  # Network architecture config
        
        if objective_fn is None:
            self.objective_fn = ObjectiveFunction()
        else:
            self.objective_fn = objective_fn
        
        self.n_evaluations = 0
        self.history = []
    
    def _objective(self, trial) -> float:
        """Optuna objective function."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Use BayesianOptimizer instead.")
        
        self.n_evaluations += 1
        
        # Sample parameters
        params = {}
        for spec in self.param_space.specs:
            if spec.log_scale:
                params[spec.name] = trial.suggest_float(
                    spec.name, spec.min_val, spec.max_val, log=True)
            else:
                params[spec.name] = trial.suggest_float(
                    spec.name, spec.min_val, spec.max_val)
        
        # Evaluate
        try:
            eval_seed = self.config.seed + self.n_evaluations if self.config.seed else self.n_evaluations
            spike_data = run_simulation(params, network_config=self.network_config, seed=eval_seed)
            fitness = self.objective_fn(spike_data)
        except Exception:
            fitness = 1e6
        
        return fitness
    
    def optimize(self) -> BOResult:
        """Run Optuna optimization."""
        try:
            import optuna
            optuna.logging.set_verbosity(
                optuna.logging.INFO if self.config.verbose else optuna.logging.WARNING)
        except ImportError:
            print("Optuna not installed. Falling back to custom BO.")
            bo = BayesianOptimizer(self.param_space, self.objective_fn, self.config, 
                                   network_config=self.network_config)
            return bo.optimize()
        
        start_time = time.time()
        
        # Create study
        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        # Optimize
        study.optimize(self._objective, n_trials=self.config.n_iterations)
        
        runtime = time.time() - start_time
        
        # Extract results
        best_params = study.best_params
        best_fitness = study.best_value
        
        # Build history
        history = []
        best_so_far = float('inf')
        for i, trial in enumerate(study.trials):
            if trial.value is not None and trial.value < best_so_far:
                best_so_far = trial.value
            history.append({
                'iteration': i,
                'best_fitness': best_so_far,
                'current_fitness': trial.value or 1e6,
            })
        
        if self.config.verbose:
            print(f"\nOptimization complete!")
            print(f"  Best fitness: {best_fitness:.4f}")
            print(f"  Runtime: {runtime:.1f}s")
        
        return BOResult(
            best_params=best_params,
            best_fitness=float(best_fitness),
            history=history,
            runtime_seconds=runtime,
            n_evaluations=self.n_evaluations,
        )


# Convenience functions
def run_bayesian_optimization(param_space: Optional[ParameterSpace] = None,
                              n_iterations: int = 50,
                              n_initial: int = 10,
                              seed: Optional[int] = None,
                              use_optuna: bool = False,
                              network_config = None) -> BOResult:
    """Run Bayesian Optimization."""
    config = BOConfig(
        n_iterations=n_iterations,
        n_initial=n_initial,
        seed=seed,
    )
    
    if use_optuna:
        optimizer = OptunaOptimizer(param_space, config=config, network_config=network_config)
    else:
        optimizer = BayesianOptimizer(param_space, config=config, network_config=network_config)
    
    return optimizer.optimize()


if __name__ == "__main__":
    # Quick test
    from core.parameters import get_reduced_parameter_space
    
    print("Testing Bayesian Optimization with reduced parameter space...")
    result = run_bayesian_optimization(
        param_space=get_reduced_parameter_space(),
        n_iterations=20,
        n_initial=5,
        seed=42
    )
    
    print(f"\nBest parameters found:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value:.6f}")