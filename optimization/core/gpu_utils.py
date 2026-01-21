"""GPU and CoreNEURON utility functions."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os


def check_coreneuron_available() -> bool:
    try:
        from neuron import coreneuron
        return True
    except ImportError:
        return False


def check_gpu_available() -> Tuple[bool, str]:
    if not check_coreneuron_available():
        return False, "CoreNEURON not installed"
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            return True, f"Found {len(gpus)} GPU(s): {', '.join(gpus)}"
        return False, "nvidia-smi failed - no CUDA GPUs found"
    except FileNotFoundError:
        return False, "nvidia-smi not found - CUDA drivers may not be installed"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"
    except Exception as e:
        return False, f"GPU check failed: {e}"


def get_gpu_info() -> Dict:
    info = {'coreneuron_available': check_coreneuron_available(),
            'gpu_available': False, 'gpu_count': 0, 'gpus': [], 'message': ''}
    
    gpu_available, message = check_gpu_available()
    info['gpu_available'] = gpu_available
    info['message'] = message
    
    if gpu_available:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        info['gpus'].append({'index': int(parts[0]), 'name': parts[1], 'memory': parts[2]})
                info['gpu_count'] = len(info['gpus'])
        except Exception:
            pass
    return info


def parse_spike_file(filepath: str) -> Dict[str, np.ndarray]:
    spikes_by_gid = {}
    filepath = Path(filepath)
    if not filepath.exists():
        return spikes_by_gid
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    time, gid = float(parts[0]), int(parts[1])
                    if gid not in spikes_by_gid:
                        spikes_by_gid[gid] = []
                    spikes_by_gid[gid].append(time)
                except ValueError:
                    continue
    
    for gid in spikes_by_gid:
        spikes_by_gid[gid] = np.array(sorted(spikes_by_gid[gid]))
    return spikes_by_gid


def aggregate_spikes_by_layer(spikes_by_gid: Dict[int, np.ndarray],
                              gid_to_layer: Dict[int, str]) -> Dict[str, np.ndarray]:
    layer_spikes = {}
    for gid, times in spikes_by_gid.items():
        if gid in gid_to_layer:
            layer = gid_to_layer[gid]
            if layer not in layer_spikes:
                layer_spikes[layer] = []
            layer_spikes[layer].extend(times.tolist())
    
    for layer in layer_spikes:
        layer_spikes[layer] = np.array(sorted(layer_spikes[layer]))
    return layer_spikes


def cleanup_coreneuron_data(datadir: str):
    datadir = Path(datadir)
    if datadir.exists() and datadir.is_dir():
        import shutil
        try:
            shutil.rmtree(datadir)
        except Exception as e:
            print(f"Warning: Could not clean up CoreNEURON data: {e}")


def estimate_gpu_memory_requirement(n_neurons: int, n_synapses: int, tstop: float, dt: float) -> float:
    """Rough estimate of GPU memory requirement in MB."""
    per_neuron_mb = 0.01
    per_synapse_mb = 0.001
    spike_buffer_mb = n_neurons * 0.001
    total_mb = n_neurons * per_neuron_mb + n_synapses * per_synapse_mb + spike_buffer_mb
    return total_mb * 1.5


def select_gpu_device(preferred: int = 0) -> int:
    info = get_gpu_info()
    if not info['gpu_available']:
        return 0
    return preferred if preferred < info['gpu_count'] else 0


def is_gpu_ready() -> bool:
    return check_coreneuron_available() and check_gpu_available()[0]


if __name__ == "__main__":
    print("GPU Status Check\n" + "=" * 40)
    info = get_gpu_info()
    print(f"CoreNEURON available: {info['coreneuron_available']}")
    print(f"GPU available: {info['gpu_available']}")
    print(f"Message: {info['message']}")
    if info['gpus']:
        print("\nAvailable GPUs:")
        for gpu in info['gpus']:
            print(f"  [{gpu['index']}] {gpu['name']} ({gpu['memory']})")
