"""
Parameter space definition for optimization.
"""

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
    log_scale: bool = False  # Whether to optimize in log space
    description: str = ""
    
    def sample_uniform(self) -> float:
        """Sample uniformly from parameter range."""
        if self.log_scale:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            return 10 ** np.random.uniform(log_min, log_max)
        return np.random.uniform(self.min_val, self.max_val)
    
    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1]."""
        if self.log_scale:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            return (np.log10(value) - log_min) / (log_max - log_min)
        return (value - self.min_val) / (self.max_val - self.min_val)
    
    def denormalize(self, norm_value: float) -> float:
        """Convert [0, 1] back to original scale."""
        norm_value = np.clip(norm_value, 0, 1)
        if self.log_scale:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            return 10 ** (log_min + norm_value * (log_max - log_min))
        return self.min_val + norm_value * (self.max_val - self.min_val)


@dataclass
class ParameterSpace:
    """
    Full parameter space for optimization.
    """
    specs: List[ParameterSpec] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.specs:
            self.specs = self._default_specs()
    
    def _default_specs(self) -> List[ParameterSpec]:
        """Default parameter specifications for thalamo-cortical model."""
        return [
            # Excitatory synapse parameters
            ParameterSpec(
                name='exc_weight_mean',
                min_val=0.0001, max_val=0.01, default=0.001,
                log_scale=True,
                description='Mean excitatory synaptic weight'
            ),
            ParameterSpec(
                name='exc_weight_std',
                min_val=0.00005, max_val=0.005, default=0.0009,
                log_scale=True,
                description='Std of excitatory synaptic weight'
            ),
            ParameterSpec(
                name='exc_tau',
                min_val=0.5, max_val=8.0, default=2.0,
                description='Excitatory synapse time constant (ms)'
            ),
            ParameterSpec(
                name='exc_delay_mean',
                min_val=0.5, max_val=8.0, default=3.0,
                description='Mean excitatory synaptic delay (ms)'
            ),
            ParameterSpec(
                name='exc_delay_std',
                min_val=0.1, max_val=4.0, default=2.0,
                description='Std of excitatory synaptic delay (ms)'
            ),
            
            # Inhibitory synapse parameters
            ParameterSpec(
                name='inh_weight',
                min_val=0.0001, max_val=0.01, default=0.001,
                log_scale=True,
                description='Inhibitory synaptic weight'
            ),
            ParameterSpec(
                name='inh_tau',
                min_val=1.0, max_val=10.0, default=3.0,
                description='Inhibitory synapse time constant (ms)'
            ),
            ParameterSpec(
                name='inh_delay',
                min_val=0.5, max_val=5.0, default=2.0,
                description='Inhibitory synaptic delay (ms)'
            ),
            ParameterSpec(
                name='inh_e',
                min_val=-85.0, max_val=-55.0, default=-75.0,
                description='Inhibitory reversal potential (mV)'
            ),
            
            # Stimulus parameters
            ParameterSpec(
                name='stim_weight',
                min_val=0.001, max_val=0.05, default=0.01,
                log_scale=True,
                description='Stimulus synaptic weight'
            ),
            ParameterSpec(
                name='stim_interval',
                min_val=5.0, max_val=30.0, default=15.0,
                description='Stimulus interval (ms)'
            ),
            ParameterSpec(
                name='stim_noise',
                min_val=0.0, max_val=1.0, default=1.0,
                description='Stimulus noise (0=regular, 1=Poisson)'
            ),
        ]
    
    @property
    def n_params(self) -> int:
        return len(self.specs)
    
    @property
    def names(self) -> List[str]:
        return [s.name for s in self.specs]
    
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """Return bounds as list of (min, max) tuples."""
        return [(s.min_val, s.max_val) for s in self.specs]
    
    @property
    def bounds_normalized(self) -> List[Tuple[float, float]]:
        """Normalized bounds (always [0, 1])."""
        return [(0.0, 1.0) for _ in self.specs]
    
    def get_default(self) -> Dict[str, float]:
        """Return default parameter values."""
        return {s.name: s.default for s in self.specs}
    
    def sample_random(self) -> Dict[str, float]:
        """Sample random parameters uniformly."""
        return {s.name: s.sample_uniform() for s in self.specs}
    
    def sample_random_n(self, n: int) -> List[Dict[str, float]]:
        """Sample n random parameter sets."""
        return [self.sample_random() for _ in range(n)]
    
    def normalize(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to normalized array."""
        return np.array([
            self.specs[i].normalize(params[self.specs[i].name])
            for i in range(self.n_params)
        ])
    
    def denormalize(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert normalized array to parameter dict."""
        return {
            self.specs[i].name: self.specs[i].denormalize(arr[i])
            for i in range(self.n_params)
        }
    
    def to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array (original scale)."""
        return np.array([params[s.name] for s in self.specs])
    
    def from_array(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dict (original scale)."""
        return {self.specs[i].name: arr[i] for i in range(self.n_params)}
    
    def clip(self, params: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to valid bounds."""
        return {
            s.name: np.clip(params[s.name], s.min_val, s.max_val)
            for s in self.specs
        }
    
    def get_spec(self, name: str) -> Optional[ParameterSpec]:
        """Get spec by parameter name."""
        for s in self.specs:
            if s.name == name:
                return s
        return None


# Convenience function to get default parameter space
def get_parameter_space() -> ParameterSpace:
    return ParameterSpace()


# Reduced parameter space for quick experiments
def get_reduced_parameter_space() -> ParameterSpace:
    """Smaller parameter space with only most important parameters."""
    specs = [
        ParameterSpec(
            name='exc_weight_mean',
            min_val=0.0003, max_val=0.005, default=0.001,
            log_scale=True,
            description='Mean excitatory synaptic weight'
        ),
        ParameterSpec(
            name='exc_tau',
            min_val=1.0, max_val=5.0, default=2.0,
            description='Excitatory synapse time constant (ms)'
        ),
        ParameterSpec(
            name='inh_weight',
            min_val=0.0003, max_val=0.005, default=0.001,
            log_scale=True,
            description='Inhibitory synaptic weight'
        ),
        ParameterSpec(
            name='inh_tau',
            min_val=2.0, max_val=6.0, default=3.0,
            description='Inhibitory synapse time constant (ms)'
        ),
        ParameterSpec(
            name='stim_weight',
            min_val=0.005, max_val=0.03, default=0.01,
            log_scale=True,
            description='Stimulus synaptic weight'
        ),
    ]
    return ParameterSpace(specs=specs)