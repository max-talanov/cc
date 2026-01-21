"""
Data loader for experimental recordings from MATLAB files.
Handles LFP data and spike times with variable-length vectors.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings


# Default mapping from channel indices to layer names
# Channel numbering is 1-based in MATLAB, 0-based here
DEFAULT_CHANNEL_TO_LAYER = {
    # L2/3: channels 1,2,3,12,13 (0-indexed: 0,1,2,11,12)
    0: 'L23', 1: 'L23', 2: 'L23', 11: 'L23', 12: 'L23',
    # L4: channels 4,16 (0-indexed: 3,15)
    3: 'L4', 15: 'L4',
    # L5: channels 5,6 (0-indexed: 4,5)
    4: 'L5', 5: 'L5',
    # L5/6 (deep layers): channels 7,8,9 (0-indexed: 6,7,8)
    6: 'L5', 7: 'L5', 8: 'L5',
    # L6: channel 10 (0-indexed: 9)
    9: 'L6',
    # Thalamus: channels 14,15 area (0-indexed: 13,14)
    10: 'L6', 13: 'thalamus', 14: 'thalamus',
}


@dataclass
class ExperimentalData:
    """
    Container for experimental recordings from MATLAB.
    """
    lfp: np.ndarray  # (n_timepoints, n_channels, n_trials)
    spike_times: Dict[str, List[np.ndarray]]  # layer -> [trial1_spikes, trial2_spikes, ...]
    sampling_rate: float = 1000.0
    n_trials: int = 127
    n_channels: int = 16
    duration_ms: float = 200.0
    channel_to_layer: Dict[int, str] = field(default_factory=lambda: DEFAULT_CHANNEL_TO_LAYER.copy())
    
    @classmethod
    def from_matlab(cls, 
                    mat_path: Union[str, Path], 
                    channel_to_layer: Optional[Dict[int, str]] = None,
                    sampling_rate: float = 1000.0,
                    verbose: bool = True) -> 'ExperimentalData':
        """
        Load experimental data from MATLAB .mat file.
        Matches parameters of analysis_mat_file.py (squeeze_me=False, struct_as_record=True).
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required for loading MATLAB files.")
        
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(f"MATLAB file not found: {mat_path}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Reading from {mat_path.name}")
            print(f"{'='*60}")
        
        # Load MATLAB data
        # squeeze_me=False: preserves dimensions (16x52)
        # struct_as_record=True: loads structs as numpy arrays (avoids mat_struct errors)
        data = loadmat(str(mat_path), squeeze_me=False, struct_as_record=True)
        
        if 'lfp' not in data:
            raise ValueError("MATLAB file must contain 'lfp' variable")
        
        lfp = data['lfp']
        
        # Robust LFP dimension handling
        # We want (n_timepoints, n_channels, n_trials)
        while lfp.ndim > 3:
            if lfp.shape[-1] == 1: 
                lfp = lfp[..., 0]
            elif lfp.shape[2] == 1 and lfp.shape[3] > 1:
                lfp = lfp[:, :, 0, :]
            else:
                lfp = np.squeeze(lfp)
                break
                
        if lfp.ndim == 2:
            lfp = lfp[:, :, np.newaxis]
            
        n_timepoints, n_channels, n_trials = lfp.shape
        duration_ms = n_timepoints / sampling_rate * 1000.0
        
        if verbose:
            print(f"  LFP Shape: {lfp.shape}")
            print(f"  Trials detected: {n_trials}")
        
        if channel_to_layer is None:
            channel_to_layer = DEFAULT_CHANNEL_TO_LAYER.copy()
        
        # Extract spike times
        if 'spks' in data:
            spike_times = cls._parse_spks_struct(
                data['spks'], 
                n_channels, 
                n_trials, 
                channel_to_layer,
                verbose=verbose
            )
        else:
            if verbose: print("  Warning: 'spks' variable not found.")
            spike_times = {layer: [np.array([]) for _ in range(n_trials)] 
                          for layer in set(channel_to_layer.values())}
        
        result = cls(
            lfp=lfp,
            spike_times=spike_times,
            sampling_rate=sampling_rate,
            n_trials=n_trials,
            n_channels=n_channels,
            duration_ms=duration_ms,
            channel_to_layer=channel_to_layer
        )
        
        if verbose:
            print(f"\n  Read complete")
            print(f"  Spike counts per layer:")
            counts = result.get_mean_spike_counts()
            for layer, count in sorted(counts.items()):
                print(f"    - {layer}: {count:.1f} spikes (mean per trial)")
            print(f"{'='*60}\n")
        
        return result
    
    @staticmethod
    def _parse_spks_struct(spks, n_channels: int, n_trials: int,
                           channel_to_layer: Dict[int, str],
                           verbose: bool = False) -> Dict[str, List[np.ndarray]]:
        """
        Parse MATLAB spks array.
        Compatible with both cell arrays and struct arrays.
        """
        layers = set(channel_to_layer.values())
        spike_times = {layer: [[] for _ in range(n_trials)] for layer in layers}
        
        if not isinstance(spks, np.ndarray):
            return spike_times
            
        actual_channels = spks.shape[0]
        
        if verbose:
            print(f"  Parsing 'spks': shape={spks.shape}, dtype={spks.dtype}")
            
        total_spikes = 0
        
        # Iterate channels (rows)
        for channel in range(min(actual_channels, n_channels)):
            layer = channel_to_layer.get(channel, 'L5')
            items = spks[channel] # This is a row of trials
            
            # Handle if items is not iterable (single trial)
            if not isinstance(items, np.ndarray):
                items = np.array([items])
                
            iter_trials = min(items.size, n_trials)
            
            for trial in range(iter_trials):
                try:
                    varb = items[trial]
                    spk_times = []
                    
                    # Try to unwrap like a cell array (access [0])
                    # This matches 'if any(varb[0])' from analysis_mat_file.py
                    try:
                        # Check if varb is indexable/has size before accessing [0]
                        # This avoids the 'mat_struct has no attribute size' error
                        # because we are now using struct_as_record=True, so structs are np.void
                        # and cells are np.ndarray
                        if isinstance(varb, np.ndarray) and varb.size > 0:
                            spk_times = _extract_spike_times(varb[0])
                        elif isinstance(varb, np.void) and len(varb) > 0:
                            # Handle structured array (struct) -> access first field
                            spk_times = _extract_spike_times(varb[0])
                        else:
                            # Direct extraction fallback
                            spk_times = _extract_spike_times(varb)
                            
                    except (IndexError, TypeError, ValueError):
                        # Fallback if [0] fails
                        spk_times = _extract_spike_times(varb)
                    
                    if spk_times:
                        spike_times[layer][trial].extend(spk_times)
                        total_spikes += len(spk_times)
                                 
                except Exception as e:
                    if verbose and trial == 0:
                        print(f"    Warn: Parse error Ch{channel} Tr{trial}: {e}")
                    continue

        # Convert to sorted numpy arrays
        for layer in spike_times:
            for trial in range(len(spike_times[layer])):
                times = spike_times[layer][trial]
                spike_times[layer][trial] = np.sort(np.unique(np.array(times)))
        
        if verbose:
            print(f"  Total spikes extracted: {total_spikes}")
            
        return spike_times

    # ... [Standard accessors below] ...
    
    def get_trial_spikes(self, trial_idx: int) -> Dict[str, np.ndarray]:
        if trial_idx < 0 or trial_idx >= self.n_trials:
            raise IndexError(f"Trial index {trial_idx} out of range [0, {self.n_trials})")
        return {layer: self.spike_times[layer][trial_idx] for layer in self.spike_times}
    
    def get_trial_lfp(self, trial_idx: int) -> np.ndarray:
        if trial_idx < 0 or trial_idx >= self.n_trials:
            raise IndexError(f"Trial index {trial_idx} out of range [0, {self.n_trials})")
        return self.lfp[:, :, trial_idx]
    
    def get_mean_lfp(self) -> np.ndarray:
        return np.mean(self.lfp, axis=2)
    
    def get_mean_spike_counts(self) -> Dict[str, float]:
        counts = {}
        for layer, trials in self.spike_times.items():
            all_counts = [len(t) for t in trials]
            counts[layer] = np.mean(all_counts) if all_counts else 0.0
        return counts
    
    def get_mean_firing_rates(self) -> Dict[str, float]:
        counts = self.get_mean_spike_counts()
        duration_s = self.duration_ms / 1000.0
        return {layer: count / duration_s for layer, count in counts.items()}
    
    def get_isis(self, layer: Optional[str] = None, trial_idx: Optional[int] = None, min_isi: float = 1.0) -> np.ndarray:
        all_isis = []
        layers_to_process = [layer] if layer else list(self.spike_times.keys())
        for lyr in layers_to_process:
            if lyr not in self.spike_times: continue
            trials = [self.spike_times[lyr][trial_idx]] if trial_idx is not None else self.spike_times[lyr]
            for spikes in trials:
                if len(spikes) > 1:
                    isis = np.diff(np.sort(spikes))
                    isis = isis[isis >= min_isi]
                    all_isis.extend(isis.tolist())
        return np.array(all_isis)
    
    def get_all_spike_times(self, layer: Optional[str] = None, trial_idx: Optional[int] = None) -> np.ndarray:
        all_spikes = []
        layers_to_process = [layer] if layer else list(self.spike_times.keys())
        for lyr in layers_to_process:
            if lyr not in self.spike_times: continue
            trials = [self.spike_times[lyr][trial_idx]] if trial_idx is not None else self.spike_times[lyr]
            for spikes in trials:
                all_spikes.extend(spikes.tolist())
        return np.sort(np.array(all_spikes))
    
    def get_layer_lfp(self, layer: str, trial_idx: Optional[int] = None) -> np.ndarray:
        layer_channels = [ch for ch, lyr in self.channel_to_layer.items() if lyr == layer]
        if not layer_channels: return np.array([])
        if trial_idx is not None:
            lfp_data = self.lfp[:, layer_channels, trial_idx]
        else:
            lfp_data = np.mean(self.lfp[:, layer_channels, :], axis=2)
        return np.mean(lfp_data, axis=1)
    
    def summary(self) -> str:
        lines = [
            "Experimental Data Summary:",
            f"  LFP shape: {self.lfp.shape} (timepoints x channels x trials)",
            f"  Duration: {self.duration_ms:.1f} ms",
            f"  Sampling rate: {self.sampling_rate} Hz",
            f"  N trials: {self.n_trials}",
            f"  N channels: {self.n_channels}",
            "",
            "  Mean spike counts per layer:",
        ]
        counts = self.get_mean_spike_counts()
        for layer, count in sorted(counts.items()):
            lines.append(f"    {layer}: {count:.1f}")
        return '\n'.join(lines)


def _extract_spike_times(cell) -> List[float]:
    """Extract spike times from a MATLAB element (recursive)."""
    if cell is None:
        return []
    
    # Case 1: Numeric scalar
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return [] if np.isnan(cell) else [float(cell)]
    
    # Case 2: Numpy Array
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return []
        # If object array (nested), recurse
        if cell.dtype == object:
            spikes = []
            for item in cell.flatten():
                spikes.extend(_extract_spike_times(item))
            return spikes
        # If numeric array, flatten
        return [float(x) for x in cell.flatten() if not np.isnan(x)]
        
    # Case 3: List/Tuple
    if isinstance(cell, (list, tuple)):
        spikes = []
        for item in cell:
            spikes.extend(_extract_spike_times(item))
        return spikes

    return []

def create_channel_mapping(n_channels: int = 16, layer_distribution: Optional[Dict[str, int]] = None) -> Dict[int, str]:
    if layer_distribution is None:
        layer_distribution = {
            'thalamus': max(1, n_channels // 6),
            'L4': max(1, n_channels // 5),
            'L23': max(1, n_channels // 5),
            'L5': max(1, n_channels // 4),
            'L6': max(1, n_channels // 6),
        }
    mapping = {}
    ch_idx = 0
    for layer, count in layer_distribution.items():
        for _ in range(count):
            if ch_idx < n_channels:
                mapping[ch_idx] = layer
                ch_idx += 1
    while ch_idx < n_channels:
        mapping[ch_idx] = 'L5'
        ch_idx += 1
    return mapping

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         ExperimentalData.from_matlab(sys.argv[1])

if __name__ == "__main__":
    # Test logic
    # import sys
    # if len(sys.argv) > 1:
    ExperimentalData.from_matlab("/home/ailab_user/tmp/cibm/tmp/cc/data/2011_may_03_P32_BCX_rust/2011_05_03_0007.mat")