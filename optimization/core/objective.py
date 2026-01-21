"""Objective functions for thalamo-cortical simulation evaluation."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .data_loader import ExperimentalData


@dataclass
class TargetBehavior:
    """Target behavior based on biological cortical activation sequence."""
    layer_latencies: Dict[str, Tuple[float, float]] = None
    layer_rates: Dict[str, Tuple[float, float]] = None
    ei_ratio: Tuple[float, float] = (2.0, 4.0)
    
    def __post_init__(self):
        if self.layer_latencies is None:
            self.layer_latencies = {
                'thalamus': (0, 10), 'L4': (8, 18), 'L23': (15, 30),
                'L5': (20, 40), 'L6': (25, 50),
            }
        if self.layer_rates is None:
            self.layer_rates = {
                'thalamus': (5, 30), 'L4': (5, 25), 'L23': (2, 15),
                'L5': (3, 20), 'L6': (2, 15),
            }


class ObjectiveFunction:
    """Computes fitness score from spike data. Lower = better."""
    
    def __init__(self, target: Optional[TargetBehavior] = None, 
                 weights: Optional[Dict[str, float]] = None):
        self.target = target or TargetBehavior()
        self.weights = weights or {
            'latency_sequence': 1.0, 'latency_timing': 0.5,
            'firing_rate': 0.3, 'activity': 0.2,
        }
    
    def __call__(self, spike_data: Dict[str, np.ndarray]) -> float:
        return self.compute(spike_data)
    
    def compute(self, spike_data: Dict[str, np.ndarray]) -> float:
        scores = self.compute_all(spike_data)
        return sum(self.weights.get(name, 0) * scores.get(name, 0) for name in self.weights)
    
    def compute_all(self, spike_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {
            'latency_sequence': self._latency_sequence_error(spike_data),
            'latency_timing': self._latency_timing_error(spike_data),
            'firing_rate': self._firing_rate_error(spike_data),
            'activity': self._activity_error(spike_data),
        }
    
    def _get_first_spike_latency(self, spikes: np.ndarray) -> float:
        return spikes[0] if len(spikes) > 0 else np.inf
    
    def _get_median_latency(self, spikes: np.ndarray, window: float = 50) -> float:
        if len(spikes) == 0:
            return np.inf
        early_spikes = spikes[spikes < window]
        return np.median(early_spikes) if len(early_spikes) > 0 else np.inf
    
    def _latency_sequence_error(self, spike_data: Dict[str, np.ndarray]) -> float:
        """Penalizes violations of: Thalamus → L4 → L2/3 → L5 → L6"""
        expected_order = ['thalamus', 'L4', 'L23', 'L5', 'L6']
        latencies = {layer: self._get_first_spike_latency(spike_data.get(layer, np.array([])))
                    for layer in expected_order}
        
        error = 0.0
        for i in range(len(expected_order) - 1):
            lat_a, lat_b = latencies[expected_order[i]], latencies[expected_order[i + 1]]
            if lat_b < lat_a:
                error += (lat_a - lat_b) ** 2
            if abs(lat_b - lat_a) < 2.0 and lat_a < np.inf:
                error += 1.0
        return error
    
    def _latency_timing_error(self, spike_data: Dict[str, np.ndarray]) -> float:
        error = 0.0
        for layer, (min_lat, max_lat) in self.target.layer_latencies.items():
            if layer not in spike_data:
                error += 100.0
                continue
            latency = self._get_median_latency(spike_data[layer])
            if latency == np.inf:
                error += 50.0
            elif latency < min_lat:
                error += (min_lat - latency) ** 2
            elif latency > max_lat:
                error += (latency - max_lat) ** 2
        return error
    
    def _firing_rate_error(self, spike_data: Dict[str, np.ndarray], duration: float = 200.0) -> float:
        error = 0.0
        for layer, (min_rate, max_rate) in self.target.layer_rates.items():
            if layer not in spike_data:
                error += 10.0
                continue
            rate = len(spike_data[layer]) / (duration / 1000.0)
            if rate < min_rate:
                error += (min_rate - rate) ** 2 / 100
            elif rate > max_rate:
                error += (rate - max_rate) ** 2 / 100
        return error
    
    def _activity_error(self, spike_data: Dict[str, np.ndarray]) -> float:
        error = 0.0
        total_spikes = sum(len(s) for s in spike_data.values())
        
        if total_spikes == 0:
            error += 1000.0
        elif total_spikes > 5000:
            error += (total_spikes - 5000) / 100
        
        for layer, spikes in spike_data.items():
            if len(spikes) == 0 and total_spikes > 10:
                error += 20.0
        return error


class MultiObjective:
    """Multi-objective evaluation returning vector of objectives."""
    
    def __init__(self, target: Optional[TargetBehavior] = None):
        self.target = target or TargetBehavior()
        self.single_obj = ObjectiveFunction(target)
    
    def __call__(self, spike_data: Dict[str, np.ndarray]) -> np.ndarray:
        scores = self.single_obj.compute_all(spike_data)
        return np.array([scores['latency_sequence'], scores['latency_timing'],
                        scores['firing_rate'], scores['activity']])
    
    @property
    def n_objectives(self) -> int:
        return 4
    
    @property
    def objective_names(self) -> List[str]:
        return ['latency_sequence', 'latency_timing', 'firing_rate', 'activity']


class SupervisedObjective:
    """Compares simulation to experimental recordings using KS tests."""
    
    def __init__(self, experimental_data: 'ExperimentalData', trial_idx: Optional[int] = None,
                 weights: Optional[Dict[str, float]] = None, duration_ms: float = 200.0):
        self.data = experimental_data
        self.trial_idx = trial_idx
        self.duration_ms = duration_ms
        self.weights = weights or {
            'isi_ks': 0.35, 'spike_ks': 0.25, 'spike_count': 0.2,
            'firing_rate': 0.1, 'lfp_correlation': 0.1,
        }
        self._target_spike_counts = self._compute_target_spike_counts()
        self._target_firing_rates = self._compute_target_firing_rates()
        self._target_spike_histograms = self._compute_target_histograms()
        self._target_isis = self._compute_target_isis()
        self._target_spike_times = self._compute_target_spike_times()
    
    def _compute_target_isis(self) -> Dict[str, np.ndarray]:
        return {layer: self.data.get_isis(layer=layer, trial_idx=self.trial_idx)
                for layer in self.data.spike_times}
    
    def _compute_target_spike_times(self) -> Dict[str, np.ndarray]:
        return {layer: self.data.get_all_spike_times(layer=layer, trial_idx=self.trial_idx)
                for layer in self.data.spike_times}
    
    def _compute_target_spike_counts(self) -> Dict[str, float]:
        if self.trial_idx is not None:
            return {layer: len(self.data.spike_times[layer][self.trial_idx])
                   for layer in self.data.spike_times}
        return self.data.get_mean_spike_counts()
    
    def _compute_target_firing_rates(self) -> Dict[str, float]:
        if self.trial_idx is not None:
            duration_s = self.data.duration_ms / 1000.0
            return {layer: count / duration_s for layer, count in self._target_spike_counts.items()}
        return self.data.get_mean_firing_rates()
    
    def _compute_target_histograms(self, bin_size_ms: float = 10.0) -> Dict[str, np.ndarray]:
        histograms = {}
        bins = np.arange(0, self.data.duration_ms + bin_size_ms, bin_size_ms)
        for layer in self.data.spike_times:
            if self.trial_idx is not None:
                spikes = self.data.spike_times[layer][self.trial_idx]
            else:
                spikes = np.concatenate(self.data.spike_times[layer])
            hist, _ = np.histogram(spikes, bins=bins, density=True)
            histograms[layer] = hist
        return histograms
    
    def __call__(self, spike_data: Dict[str, np.ndarray], 
                 lfp_data: Optional[np.ndarray] = None) -> float:
        loss = 0.0
        if self.weights.get('isi_ks', 0) > 0:
            loss += self.weights['isi_ks'] * self._isi_ks_loss(spike_data)
        if self.weights.get('spike_ks', 0) > 0:
            loss += self.weights['spike_ks'] * self._spike_ks_loss(spike_data)
        if self.weights.get('spike_count', 0) > 0:
            loss += self.weights['spike_count'] * self._spike_count_loss(spike_data)
        if self.weights.get('firing_rate', 0) > 0:
            loss += self.weights['firing_rate'] * self._firing_rate_loss(spike_data)
        if lfp_data is not None and self.weights.get('lfp_correlation', 0) > 0:
            loss += self.weights['lfp_correlation'] * self._lfp_loss(lfp_data)
        return loss
    
    def _isi_ks_loss(self, spike_data: Dict[str, np.ndarray]) -> float:
        from scipy.stats import ks_2samp
        total_loss, n_layers = 0.0, 0
        
        for layer in self._target_isis:
            target_isis = self._target_isis[layer]
            if layer not in spike_data or len(spike_data[layer]) < 2:
                total_loss += 3.0
                n_layers += 1
                continue
            
            sim_spikes = np.sort(spike_data[layer])
            sim_isis = np.diff(sim_spikes)
            sim_isis = sim_isis[sim_isis >= 1.0]
            
            if len(target_isis) < 2 or len(sim_isis) < 2:
                total_loss += 3.0
                n_layers += 1
                continue
            
            try:
                stat, _ = ks_2samp(sim_isis, target_isis)
                total_loss += stat
            except Exception:
                total_loss += 3.0
            n_layers += 1
        
        return (total_loss / max(n_layers, 1)) * 100
    
    def _spike_ks_loss(self, spike_data: Dict[str, np.ndarray]) -> float:
        from scipy.stats import ks_2samp
        total_loss, n_layers = 0.0, 0
        
        for layer in self._target_spike_times:
            target_spikes = self._target_spike_times[layer]
            if layer not in spike_data:
                total_loss += 3.0
                n_layers += 1
                continue
            
            sim_spikes = spike_data[layer]
            if len(target_spikes) < 2 or len(sim_spikes) < 2:
                total_loss += 3.0
                n_layers += 1
                continue
            
            try:
                stat, _ = ks_2samp(sim_spikes, target_spikes)
                total_loss += stat
            except Exception:
                total_loss += 3.0
            n_layers += 1
        
        return (total_loss / max(n_layers, 1)) * 100
    
    def _spike_count_loss(self, spike_data: Dict[str, np.ndarray]) -> float:
        """Log-MSE to prevent silence trap."""
        loss, n_layers = 0.0, 0
        for layer in self._target_spike_counts:
            if layer not in spike_data:
                loss += 100.0
                continue
            
            target_count = self._target_spike_counts[layer]
            sim_count = len(spike_data[layer])
            
            if target_count > 0:
                log_diff = np.log(sim_count + 1) - np.log(target_count + 1)
                loss += (log_diff ** 2) * 20.0
                if sim_count == 0:
                    loss += 50.0
            else:
                loss += sim_count / 10.0
            n_layers += 1
        
        return loss / max(n_layers, 1)
    
    def _firing_rate_loss(self, spike_data: Dict[str, np.ndarray]) -> float:
        loss, n_layers = 0.0, 0
        duration_s = self.duration_ms / 1000.0
        
        for layer in self._target_firing_rates:
            if layer not in spike_data:
                loss += 10.0
                continue
            target_rate = self._target_firing_rates[layer]
            sim_rate = len(spike_data[layer]) / duration_s
            loss += (target_rate - sim_rate) ** 2
            n_layers += 1
        
        return loss / max(n_layers, 1)
    
    def _lfp_loss(self, simulated_lfp: np.ndarray) -> float:
        from scipy.stats import ks_2samp
        
        target_lfp = (self.data.get_trial_lfp(self.trial_idx) if self.trial_idx is not None 
                      else self.data.get_mean_lfp())
        
        if simulated_lfp.ndim == 1:
            simulated_lfp = simulated_lfp[:, np.newaxis]
        if target_lfp.ndim == 1:
            target_lfp = target_lfp[:, np.newaxis]
        
        if simulated_lfp.shape != target_lfp.shape:
            min_samples = min(simulated_lfp.shape[0], target_lfp.shape[0])
            min_channels = min(simulated_lfp.shape[1], target_lfp.shape[1])
            simulated_lfp = simulated_lfp[:min_samples, :min_channels]
            target_lfp = target_lfp[:min_samples, :min_channels]
        
        sim_flat, target_flat = simulated_lfp.flatten(), target_lfp.flatten()
        if len(sim_flat) < 2 or len(target_flat) < 2:
            return 100.0
        
        try:
            ks_stat, _ = ks_2samp(sim_flat, target_flat)
        except Exception:
            ks_stat = 1.0
        
        if np.std(sim_flat) < 1e-10 or np.std(target_flat) < 1e-10:
            corr_loss = 1.0
        else:
            correlation = np.corrcoef(sim_flat, target_flat)[0, 1]
            corr_loss = 1.0 if np.isnan(correlation) else 1.0 - correlation
        
        return 0.7 * ks_stat * 100 + 0.3 * corr_loss * 100
    
    def compute_all(self, spike_data: Dict[str, np.ndarray],
                    lfp_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        components = {
            'isi_ks': self._isi_ks_loss(spike_data),
            'spike_ks': self._spike_ks_loss(spike_data),
            'spike_count': self._spike_count_loss(spike_data),
            'firing_rate': self._firing_rate_loss(spike_data),
        }
        if lfp_data is not None:
            components['lfp_correlation'] = self._lfp_loss(lfp_data)
        return components


class HybridObjective:
    """Combines rule-based and supervised objectives."""
    
    def __init__(self, experimental_data: Optional['ExperimentalData'] = None,
                 supervised_weight: float = 0.5, trial_idx: Optional[int] = None,
                 rule_based_target: Optional[TargetBehavior] = None):
        self.rule_based = ObjectiveFunction(target=rule_based_target)
        self.supervised_weight = np.clip(supervised_weight, 0.0, 1.0)
        
        if experimental_data is not None:
            self.supervised = SupervisedObjective(experimental_data=experimental_data, trial_idx=trial_idx)
        else:
            self.supervised = None
            self.supervised_weight = 0.0
    
    def __call__(self, spike_data: Dict[str, np.ndarray],
                 lfp_data: Optional[np.ndarray] = None) -> float:
        rule_loss = self.rule_based(spike_data)
        if self.supervised is not None and self.supervised_weight > 0:
            sup_loss = self.supervised(spike_data, lfp_data)
            return (1 - self.supervised_weight) * rule_loss + self.supervised_weight * sup_loss
        return rule_loss
    
    def compute_all(self, spike_data: Dict[str, np.ndarray],
                    lfp_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        components = {}
        for name, value in self.rule_based.compute_all(spike_data).items():
            components[f'rule_{name}'] = value
        if self.supervised is not None:
            for name, value in self.supervised.compute_all(spike_data, lfp_data).items():
                components[f'sup_{name}'] = value
        components['total'] = self(spike_data, lfp_data)
        return components


def evaluate_simulation(spike_data: Dict[str, np.ndarray], multi_objective: bool = False) -> float:
    if multi_objective:
        return MultiObjective()(spike_data)
    return ObjectiveFunction()(spike_data)


def create_objective(
    objective_type: str = 'rule-based',
    experimental_data: Optional['ExperimentalData'] = None,
    supervised_weight: float = 0.5,
    trial_idx: Optional[int] = None
) -> Union[ObjectiveFunction, SupervisedObjective, HybridObjective]:
    if objective_type == 'rule-based':
        return ObjectiveFunction()
    elif objective_type == 'supervised':
        if experimental_data is None:
            raise ValueError("supervised objective requires experimental_data")
        return SupervisedObjective(experimental_data=experimental_data, trial_idx=trial_idx)
    elif objective_type == 'hybrid':
        return HybridObjective(experimental_data=experimental_data,
                              supervised_weight=supervised_weight, trial_idx=trial_idx)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
