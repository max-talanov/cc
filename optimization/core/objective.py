"""
Objective functions for evaluating thalamo-cortical simulation quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TargetBehavior:

    # TODO: increase penalty for 2 spiking before 4
    # TODO: relative frequencies penalty
    """
    Target behavior for the thalamo-cortical column.
    Based on biological observations of cortical activation sequence.
    """
    layer_latencies: Dict[str, Tuple[float, float]] = None  # (min, max)
    
    layer_rates: Dict[str, Tuple[float, float]] = None  # (min, max)
    
    ei_ratio: Tuple[float, float] = (2.0, 4.0)
    
    def __post_init__(self):
        if self.layer_latencies is None:
            # Biological latencies: Thalamus → L4 → L2/3 → L5 → L6
            self.layer_latencies = {
                'thalamus': (0, 10),
                'L4': (8, 18),
                'L23': (15, 30),
                'L5': (20, 40),
                'L6': (25, 50),
            }
        
        if self.layer_rates is None:
            # Typical cortical firing rates
            self.layer_rates = {
                'thalamus': (5, 30),
                'L4': (5, 25),
                'L23': (2, 15),
                'L5': (3, 20),
                'L6': (2, 15),
            }


class ObjectiveFunction:
    """
    Computes fitness score from simulation spike data.
    Lower score = better fit to target behavior.
    """
    
    def __init__(self, target: Optional[TargetBehavior] = None, 
                 weights: Optional[Dict[str, float]] = None):
        self.target = target or TargetBehavior()
        
        # Weights for combining different objectives
        self.weights = weights or {
            'latency_sequence': 1.0,    # Correct activation order
            'latency_timing': 0.5,      # Correct absolute timing
            'firing_rate': 0.3,         # Rates in biological range
            'activity': 0.2,            # Network not silent/exploding
        }
    
    def __call__(self, spike_data: Dict[str, np.ndarray]) -> float:
        """Compute total objective (lower = better)."""
        return self.compute(spike_data)
    
    def compute(self, spike_data: Dict[str, np.ndarray]) -> float:
        """Compute weighted sum of all objectives."""
        scores = self.compute_all(spike_data)
        
        total = 0.0
        for name, weight in self.weights.items():
            if name in scores:
                total += weight * scores[name]
        
        return total
    
    def compute_all(self, spike_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute all individual objective components."""
        return {
            'latency_sequence': self._latency_sequence_error(spike_data),
            'latency_timing': self._latency_timing_error(spike_data),
            'firing_rate': self._firing_rate_error(spike_data),
            'activity': self._activity_error(spike_data),
        }
    
    def _get_first_spike_latency(self, spikes: np.ndarray) -> float:
        """Get time of first spike, or inf if no spikes."""
        if len(spikes) == 0:
            return np.inf
        return spikes[0]
    
    def _get_median_latency(self, spikes: np.ndarray, window: float = 50) -> float:
        """Get median spike time in early window."""
        if len(spikes) == 0:
            return np.inf
        early_spikes = spikes[spikes < window]
        if len(early_spikes) == 0:
            return np.inf
        return np.median(early_spikes)
    
    def _latency_sequence_error(self, spike_data: Dict[str, np.ndarray]) -> float:
        """
        Check if layers activate in correct order.
        Penalizes violations of: Thalamus → L4 → L2/3 → L5 → L6
        """
        expected_order = ['thalamus', 'L4', 'L23', 'L5', 'L6']
        
        # Get first spike latency for each layer
        latencies = {}
        for layer in expected_order:
            if layer in spike_data:
                latencies[layer] = self._get_first_spike_latency(spike_data[layer])
            else:
                latencies[layer] = np.inf
        
        # Count order violations
        error = 0.0
        for i in range(len(expected_order) - 1):
            layer_a = expected_order[i]
            layer_b = expected_order[i + 1]
            
            lat_a = latencies[layer_a]
            lat_b = latencies[layer_b]
            
            # Penalize if later layer activates before earlier layer
            if lat_b < lat_a:
                error += (lat_a - lat_b) ** 2
            
            # Small penalty for simultaneous activation (should have some delay)
            if abs(lat_b - lat_a) < 2.0 and lat_a < np.inf:
                error += 1.0
        
        return error
    
    def _latency_timing_error(self, spike_data: Dict[str, np.ndarray]) -> float:
        """
        Check if absolute latencies are in expected ranges.
        """
        error = 0.0
        
        for layer, (min_lat, max_lat) in self.target.layer_latencies.items():
            if layer not in spike_data:
                error += 100.0  # Missing layer penalty
                continue
            
            latency = self._get_median_latency(spike_data[layer])
            
            if latency == np.inf:
                error += 50.0  # No spikes penalty
            elif latency < min_lat:
                error += (min_lat - latency) ** 2
            elif latency > max_lat:
                error += (latency - max_lat) ** 2
        
        return error
    
    def _firing_rate_error(self, spike_data: Dict[str, np.ndarray], 
                          duration: float = 200.0) -> float:
        """
        Check if firing rates are in biological ranges.
        """
        error = 0.0
        
        for layer, (min_rate, max_rate) in self.target.layer_rates.items():
            if layer not in spike_data:
                error += 10.0
                continue
            
            n_spikes = len(spike_data[layer])
            rate = n_spikes / (duration / 1000.0)  # Convert to Hz
            
            if rate < min_rate:
                error += (min_rate - rate) ** 2 / 100
            elif rate > max_rate:
                error += (rate - max_rate) ** 2 / 100
        
        return error
    
    def _activity_error(self, spike_data: Dict[str, np.ndarray]) -> float:
        """
        Penalize pathological activity patterns.
        """
        error = 0.0
        
        total_spikes = sum(len(s) for s in spike_data.values())
        
        # Penalize complete silence
        if total_spikes == 0:
            error += 1000.0
        
        # Penalize explosion (too many spikes)
        elif total_spikes > 5000:
            error += (total_spikes - 5000) / 100
        
        # Penalize if any layer is silent while others are active
        for layer, spikes in spike_data.items():
            if len(spikes) == 0 and total_spikes > 10:
                error += 20.0
        
        return error


class MultiObjective:
    """
    Multi-objective evaluation for Pareto optimization.
    Returns vector of objectives instead of scalar.
    """
    
    def __init__(self, target: Optional[TargetBehavior] = None):
        self.target = target or TargetBehavior()
        self.single_obj = ObjectiveFunction(target)
    
    def __call__(self, spike_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Return array of objective values for multi-objective optimization."""
        scores = self.single_obj.compute_all(spike_data)
        return np.array([
            scores['latency_sequence'],
            scores['latency_timing'],
            scores['firing_rate'],
            scores['activity'],
        ])
    
    @property
    def n_objectives(self) -> int:
        return 4
    
    @property
    def objective_names(self) -> List[str]:
        return ['latency_sequence', 'latency_timing', 'firing_rate', 'activity']


# Convenience function
def evaluate_simulation(spike_data: Dict[str, np.ndarray], 
                       multi_objective: bool = False) -> float:
    """Evaluate simulation results."""
    if multi_objective:
        return MultiObjective()(spike_data)
    return ObjectiveFunction()(spike_data)