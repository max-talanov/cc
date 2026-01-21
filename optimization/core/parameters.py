"""Parameter space definition for optimization."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    min_val: float
    max_val: float
    default: float
    log_scale: bool = False
    description: str = ""
    
    def sample_uniform(self) -> float:
        if self.log_scale:
            return 10 ** np.random.uniform(np.log10(self.min_val), np.log10(self.max_val))
        return np.random.uniform(self.min_val, self.max_val)
    
    def normalize(self, value: float) -> float:
        if self.log_scale:
            log_min, log_max = np.log10(self.min_val), np.log10(self.max_val)
            return (np.log10(value) - log_min) / (log_max - log_min)
        return (value - self.min_val) / (self.max_val - self.min_val)
    
    def denormalize(self, norm_value: float) -> float:
        norm_value = np.clip(norm_value, 0, 1)
        if self.log_scale:
            log_min, log_max = np.log10(self.min_val), np.log10(self.max_val)
            return 10 ** (log_min + norm_value * (log_max - log_min))
        return self.min_val + norm_value * (self.max_val - self.min_val)


@dataclass
class ParameterSpace:
    """Full parameter space for optimization."""
    specs: List[ParameterSpec] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.specs:
            self.specs = self._default_specs()
    
    def _default_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec('exc_weight_mean', 0.0001, 0.01, 0.001, True, 'Mean excitatory weight'),
            ParameterSpec('exc_weight_std', 0.00005, 0.005, 0.0009, True, 'Std excitatory weight'),
            ParameterSpec('exc_tau', 0.5, 8.0, 2.0, False, 'Excitatory tau (ms)'),
            ParameterSpec('exc_delay_mean', 0.5, 8.0, 3.0, False, 'Mean excitatory delay (ms)'),
            ParameterSpec('exc_delay_std', 0.1, 4.0, 2.0, False, 'Std excitatory delay (ms)'),
            ParameterSpec('inh_weight', 0.0001, 0.01, 0.001, True, 'Inhibitory weight'),
            ParameterSpec('inh_tau', 1.0, 10.0, 3.0, False, 'Inhibitory tau (ms)'),
            ParameterSpec('inh_delay', 0.5, 5.0, 2.0, False, 'Inhibitory delay (ms)'),
            ParameterSpec('inh_e', -85.0, -55.0, -75.0, False, 'Inhibitory reversal (mV)'),
            ParameterSpec('stim_weight', 0.001, 0.05, 0.01, True, 'Stimulus weight'),
            ParameterSpec('stim_interval', 5.0, 30.0, 15.0, False, 'Stimulus interval (ms)'),
            ParameterSpec('stim_noise', 0.0, 1.0, 1.0, False, 'Stimulus noise'),
        ]
    
    @property
    def n_params(self) -> int:
        return len(self.specs)
    
    @property
    def names(self) -> List[str]:
        return [s.name for s in self.specs]
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(s.min_val, s.max_val) for s in self.specs]
    
    @property
    def bounds_normalized(self) -> List[Tuple[float, float]]:
        return [(0.0, 1.0) for _ in self.specs]
    
    def get_default(self) -> Dict[str, float]:
        return {s.name: s.default for s in self.specs}
    
    def sample_random(self) -> Dict[str, float]:
        return {s.name: s.sample_uniform() for s in self.specs}
    
    def sample_random_n(self, n: int) -> List[Dict[str, float]]:
        return [self.sample_random() for _ in range(n)]
    
    def normalize(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([self.specs[i].normalize(params[self.specs[i].name]) for i in range(self.n_params)])
    
    def denormalize(self, arr: np.ndarray) -> Dict[str, float]:
        return {self.specs[i].name: self.specs[i].denormalize(arr[i]) for i in range(self.n_params)}
    
    def to_array(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[s.name] for s in self.specs])
    
    def from_array(self, arr: np.ndarray) -> Dict[str, float]:
        return {self.specs[i].name: arr[i] for i in range(self.n_params)}
    
    def clip(self, params: Dict[str, float]) -> Dict[str, float]:
        return {s.name: np.clip(params[s.name], s.min_val, s.max_val) for s in self.specs}
    
    def get_spec(self, name: str) -> Optional[ParameterSpec]:
        for s in self.specs:
            if s.name == name:
                return s
        return None


def get_parameter_space() -> ParameterSpace:
    return ParameterSpace()


def get_reduced_parameter_space() -> ParameterSpace:
    """Smaller parameter space with only most important parameters."""
    specs = [
        ParameterSpec('exc_weight_mean', 0.0003, 0.005, 0.001, True, 'Mean excitatory weight'),
        ParameterSpec('exc_tau', 1.0, 5.0, 2.0, False, 'Excitatory tau (ms)'),
        ParameterSpec('inh_weight', 0.0003, 0.005, 0.001, True, 'Inhibitory weight'),
        ParameterSpec('inh_tau', 2.0, 6.0, 3.0, False, 'Inhibitory tau (ms)'),
        ParameterSpec('stim_weight', 0.005, 0.03, 0.01, True, 'Stimulus weight'),
    ]
    return ParameterSpace(specs=specs)
