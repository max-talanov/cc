"""
Simulator module wrapping HHNeuron thalamo-cortical column model.
Reuses code from the original notebook.
Supports configurable network architecture via YAML/JSON config.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json


# Default network configuration
DEFAULT_NETWORK_CONFIG = {
    'network': {
        'thalamus': {'E': 5, 'I': 1},
        'L4': {'E': 24, 'I': 6},
        'L23': {'E': 24, 'I': 6},
        'L5': {'E': 20, 'I': 5},
        'L6': {'E': 16, 'I': 4},
    },
    'splits': {
        'L23_E': 2,
        'L23_I': 3,
        'L4_E': 1,
        'L4_I': 1,
        'L5_E': 2,
        'L56_I': 3,
        'L6_E': 1,
    },
    'simulation': {
        'tstop': 200.0,
        'dt': 0.025,
        'v_init': -65,
    }
}


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""
    # Population sizes
    N_thalamus_E: int = 5
    N_thalamus_I: int = 1
    N_L4_E: int = 24
    N_L4_I: int = 6
    N_L23_E: int = 24
    N_L23_I: int = 6
    N_L5_E: int = 20
    N_L5_I: int = 5
    N_L6_E: int = 16
    N_L6_I: int = 4
    
    # Subgroup splits
    split_L23_E: int = 2
    split_L23_I: int = 3
    split_L4_E: int = 1
    split_L4_I: int = 1
    split_L5_E: int = 2
    split_L56_I: int = 3
    split_L6_E: int = 1
    
    # Simulation
    tstop: float = 200.0
    dt: float = 0.025
    v_init: float = -65.0
    
    @classmethod
    def from_dict(cls, config: dict) -> 'NetworkConfig':
        """Create NetworkConfig from nested dictionary."""
        kwargs = {}
        
        # Parse network populations
        if 'network' in config:
            net = config['network']
            if 'thalamus' in net:
                kwargs['N_thalamus_E'] = net['thalamus'].get('E', 5)
                kwargs['N_thalamus_I'] = net['thalamus'].get('I', 1)
            if 'L4' in net:
                kwargs['N_L4_E'] = net['L4'].get('E', 24)
                kwargs['N_L4_I'] = net['L4'].get('I', 6)
            if 'L23' in net:
                kwargs['N_L23_E'] = net['L23'].get('E', 24)
                kwargs['N_L23_I'] = net['L23'].get('I', 6)
            if 'L5' in net:
                kwargs['N_L5_E'] = net['L5'].get('E', 20)
                kwargs['N_L5_I'] = net['L5'].get('I', 5)
            if 'L6' in net:
                kwargs['N_L6_E'] = net['L6'].get('E', 16)
                kwargs['N_L6_I'] = net['L6'].get('I', 4)
        
        # Parse splits
        if 'splits' in config:
            splits = config['splits']
            kwargs['split_L23_E'] = splits.get('L23_E', 2)
            kwargs['split_L23_I'] = splits.get('L23_I', 3)
            kwargs['split_L4_E'] = splits.get('L4_E', 1)
            kwargs['split_L4_I'] = splits.get('L4_I', 1)
            kwargs['split_L5_E'] = splits.get('L5_E', 2)
            kwargs['split_L56_I'] = splits.get('L56_I', 3)
            kwargs['split_L6_E'] = splits.get('L6_E', 1)
        
        # Parse simulation
        if 'simulation' in config:
            sim = config['simulation']
            kwargs['tstop'] = sim.get('tstop', 200.0)
            kwargs['dt'] = sim.get('dt', 0.025)
            kwargs['v_init'] = sim.get('v_init', -65.0)
        
        return cls(**kwargs)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'NetworkConfig':
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            print(f"Config file not found: {path}. Using defaults.")
            return cls()
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    print("PyYAML not installed. Trying JSON fallback.")
                    return cls()
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return cls.from_dict(config)
    
    def to_dict(self) -> dict:
        """Convert to nested dictionary format."""
        return {
            'network': {
                'thalamus': {'E': self.N_thalamus_E, 'I': self.N_thalamus_I},
                'L4': {'E': self.N_L4_E, 'I': self.N_L4_I},
                'L23': {'E': self.N_L23_E, 'I': self.N_L23_I},
                'L5': {'E': self.N_L5_E, 'I': self.N_L5_I},
                'L6': {'E': self.N_L6_E, 'I': self.N_L6_I},
            },
            'splits': {
                'L23_E': self.split_L23_E,
                'L23_I': self.split_L23_I,
                'L4_E': self.split_L4_E,
                'L4_I': self.split_L4_I,
                'L5_E': self.split_L5_E,
                'L56_I': self.split_L56_I,
                'L6_E': self.split_L6_E,
            },
            'simulation': {
                'tstop': self.tstop,
                'dt': self.dt,
                'v_init': self.v_init,
            }
        }
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        config = self.to_dict()
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    yaml.dump(config, f, default_flow_style=False)
                except ImportError:
                    # Fallback to JSON
                    json.dump(config, f, indent=2)
            else:
                json.dump(config, f, indent=2)
    
    @property
    def total_neurons(self) -> int:
        """Total number of neurons in the network."""
        return (
            self.N_thalamus_E + self.N_thalamus_I +
            self.N_L4_E + self.N_L4_I +
            self.N_L23_E + self.N_L23_I +
            self.N_L5_E + self.N_L5_I +
            self.N_L6_E + self.N_L6_I
        )
    
    def summary(self) -> str:
        """Return a summary string of the network."""
        lines = [
            "Network Configuration:",
            f"  Thalamus: {self.N_thalamus_E}E / {self.N_thalamus_I}I",
            f"  L4:       {self.N_L4_E}E / {self.N_L4_I}I",
            f"  L2/3:     {self.N_L23_E}E / {self.N_L23_I}I",
            f"  L5:       {self.N_L5_E}E / {self.N_L5_I}I",
            f"  L6:       {self.N_L6_E}E / {self.N_L6_I}I",
            f"  Total:    {self.total_neurons} neurons",
        ]
        return '\n'.join(lines)


@dataclass
class SimulationParams:
    """Parameters for synaptic connections and stimulation."""
    # Excitatory synapse parameters
    exc_weight_mean: float = 0.001
    exc_weight_std: float = 0.0009
    exc_tau: float = 2.0
    exc_delay_mean: float = 3.0
    exc_delay_std: float = 2.0
    exc_e: float = 0.0
    
    # Inhibitory synapse parameters
    inh_weight: float = 0.001
    inh_tau: float = 3.0
    inh_delay: float = 2.0
    inh_e: float = -75.0
    
    # Stimulus parameters
    stim_start: float = 0.0
    stim_number: int = 20
    stim_interval: float = 15.0
    stim_noise: float = 1.0
    stim_weight: float = 0.01
    
    # Simulation duration (can be overridden by NetworkConfig)
    tstop: float = 200.0
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SimulationParams':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ThalamoCorticalSimulator:
    """
    Simulates a thalamo-cortical column with HH neurons.
    Based on the original notebook implementation.
    Supports configurable network architecture.
    """
    
    _neuron_warning_shown = False
    
    def __init__(self, 
                 params: Optional[SimulationParams] = None,
                 network_config: Optional[NetworkConfig] = None,
                 config_path: Optional[Union[str, Path]] = None):
        """
        Initialize simulator.
        
        Args:
            params: Synaptic/stimulation parameters
            network_config: Network architecture config (takes precedence)
            config_path: Path to YAML/JSON config file
        """
        self.params = params or SimulationParams()
        
        # Load network config
        if network_config is not None:
            self.network_config = network_config
        elif config_path is not None:
            self.network_config = NetworkConfig.from_file(config_path)
        else:
            self.network_config = NetworkConfig()
        
        # Override tstop from network config if not explicitly set
        if self.params.tstop == 200.0:  # default
            self.params.tstop = self.network_config.tstop
        
        self._neuron_imported = False
        self._setup_neuron()
    
    def _setup_neuron(self):
        """Import NEURON and set up simulation environment."""
        try:
            from neuron import h
            h.load_file('stdrun.hoc')
            self.h = h
            self._neuron_imported = True
        except ImportError:
            if not ThalamoCorticalSimulator._neuron_warning_shown:
                print("Note: NEURON not available. Running in mock mode.")
                ThalamoCorticalSimulator._neuron_warning_shown = True
            self.h = None
            self._neuron_imported = False
    
    def _create_hh_neuron(self, inh: bool = False):
        """Create a single HH neuron (from notebook)."""
        if not self._neuron_imported:
            return MockNeuron(inh)
        
        h = self.h
        
        class HHNeuron:
            def __init__(self, inh=False):
                self.soma = h.Section(name='soma')
                self.soma.L = 20
                self.soma.diam = 20
                self.soma.insert('hh')
                self.inh = inh
                
                self.vvec = h.Vector()
                self.vvec.record(self.soma(0.5)._ref_v)
                self.tvec = h.Vector()
                self.tvec.record(h._ref_t)
        
        return HHNeuron(inh)
    
    @staticmethod
    def flatten(population):
        """Flatten nested population lists (from notebook)."""
        if isinstance(population, list) and len(population) > 0 and isinstance(population[0], list):
            return [neuron for subgroup in population for neuron in subgroup]
        return population
    
    @staticmethod
    def split_population(population, n_subgroups):
        """Split population into subgroups (from notebook)."""
        if n_subgroups <= 0:
            return [population]
        size = len(population)
        if size == 0:
            return [[] for _ in range(n_subgroups)]
        step = max(1, size // n_subgroups)
        return [population[i*step:(i+1)*step] for i in range(n_subgroups)]
    
    def connect_exc(self, source_neurons, target_neurons) -> Tuple[list, list]:
        """Connect excitatory neurons with Gaussian weight/delay (from notebook)."""
        import random
        
        source_neurons = self.flatten(source_neurons)
        target_neurons = self.flatten(target_neurons)
        netcons = []
        synapses = []
        
        if not self._neuron_imported:
            return synapses, netcons
        
        h = self.h
        p = self.params
        
        for src in source_neurons:
            for tgt in target_neurons:
                syn = h.ExpSyn(tgt.soma(0.5))
                syn.e = p.exc_e
                syn.tau = p.exc_tau
                
                nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
                nc.threshold = 0
                
                w = max(0.0, random.gauss(p.exc_weight_mean, p.exc_weight_std))
                d = max(0.1, random.gauss(p.exc_delay_mean, p.exc_delay_std))
                
                nc.weight[0] = w
                nc.delay = d
                
                synapses.append(syn)
                netcons.append(nc)
        
        return synapses, netcons
    
    def connect_inh(self, source_neurons, target_neurons) -> Tuple[list, list]:
        """Connect inhibitory neurons (from notebook)."""
        source_neurons = self.flatten(source_neurons)
        target_neurons = self.flatten(target_neurons)
        netcons = []
        synapses = []
        
        if not self._neuron_imported:
            return synapses, netcons
        
        h = self.h
        p = self.params
        
        for src in source_neurons:
            for tgt in target_neurons:
                syn = h.ExpSyn(tgt.soma(0.5))
                syn.e = p.inh_e
                syn.tau = p.inh_tau
                
                nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
                nc.threshold = 0
                nc.weight[0] = p.inh_weight
                nc.delay = p.inh_delay
                
                synapses.append(syn)
                netcons.append(nc)
        
        return synapses, netcons
    
    def stimulate_group(self, group) -> Tuple[list, list, list]:
        """Apply stimulus to a neuron group (from notebook)."""
        syn_inputs = []
        conns = []
        netstims = []
        
        if not self._neuron_imported:
            return syn_inputs, conns, netstims
        
        h = self.h
        p = self.params
        
        for cell in self.flatten(group):
            netstim = h.NetStim()
            netstim.start = p.stim_start
            netstim.number = p.stim_number
            netstim.interval = p.stim_interval
            netstim.noise = p.stim_noise
            
            syn = h.ExpSyn(cell.soma(0.5))
            syn.e = p.exc_e
            syn.tau = p.exc_tau
            
            nc = h.NetCon(netstim, syn)
            nc.weight[0] = p.stim_weight
            
            syn_inputs.append(syn)
            conns.append(nc)
            netstims.append(netstim)
        
        return syn_inputs, conns, netstims
    
    def build_network(self) -> Dict[str, list]:
        """
        Build the full thalamo-cortical network.
        Uses network_config for population sizes.
        """
        cfg = self.network_config
        
        populations = {
            # Thalamus
            'thalamus_E': [self._create_hh_neuron(inh=False) 
                          for _ in range(cfg.N_thalamus_E)],
            'thalamus_I': [self._create_hh_neuron(inh=True) 
                          for _ in range(cfg.N_thalamus_I)],
            
            # Layer 4
            'L4_E': self.split_population(
                [self._create_hh_neuron() for _ in range(cfg.N_L4_E)], 
                cfg.split_L4_E),
            'L4_I': self.split_population(
                [self._create_hh_neuron(inh=True) for _ in range(cfg.N_L4_I)], 
                cfg.split_L4_I),
            
            # Layer 2/3
            'L23_E_RS': self.split_population(
                [self._create_hh_neuron() for _ in range(cfg.N_L23_E)], 
                cfg.split_L23_E),
            'L23_E_FRB': self.split_population(
                [self._create_hh_neuron() for _ in range(cfg.N_L23_E)], 
                cfg.split_L23_E),
            'L23_I_Bask': self.split_population(
                [self._create_hh_neuron(inh=True) for _ in range(cfg.N_L23_I)], 
                cfg.split_L23_I),
            'L23_I_LTS': self.split_population(
                [self._create_hh_neuron(inh=True) for _ in range(cfg.N_L23_I)], 
                cfg.split_L23_I),
            'L23_I_Axax': self.split_population(
                [self._create_hh_neuron(inh=True) for _ in range(cfg.N_L23_I)], 
                cfg.split_L23_I),
            
            # Layer 5
            'L5_E_RS': self.split_population(
                [self._create_hh_neuron() for _ in range(cfg.N_L5_E)], 
                cfg.split_L5_E),
            'L5_E_IB': self.split_population(
                [self._create_hh_neuron() for _ in range(cfg.N_L5_E)], 
                cfg.split_L5_E),
            
            # Layer 5/6 inhibitory (shared pool)
            'L56_I_Bask': self.split_population(
                [self._create_hh_neuron(inh=True) 
                 for _ in range(cfg.N_L5_I + cfg.N_L6_I)], 
                cfg.split_L56_I),
            'L56_I_LTS': self.split_population(
                [self._create_hh_neuron(inh=True) 
                 for _ in range(cfg.N_L5_I + cfg.N_L6_I)], 
                cfg.split_L56_I),
            'L56_I_Axax': self.split_population(
                [self._create_hh_neuron(inh=True) 
                 for _ in range(cfg.N_L5_I + cfg.N_L6_I)], 
                cfg.split_L56_I),
            
            # Layer 6
            'L6_E': self.split_population(
                [self._create_hh_neuron() for _ in range(cfg.N_L6_E)], 
                cfg.split_L6_E),
        }
        
        return populations
    
    def connect_network(self, populations: Dict[str, list]) -> Dict[str, Tuple[list, list]]:
        """Wire up the network connections (from notebook)."""
        connections = {}
        
        # Thalamus to L4
        connections['TCR_to_L4'] = self.connect_exc(
            populations['thalamus_E'], populations['L4_E'])
        connections['TCR_to_nRT'] = self.connect_exc(
            populations['thalamus_E'], populations['thalamus_I'])
        
        # L4 connections
        connections['L4_to_L4'] = self.connect_exc(
            populations['L4_E'], populations['L4_E'])
        connections['L4_to_L23_RS'] = self.connect_exc(
            populations['L4_E'], populations['L23_E_RS'])
        connections['L4_to_L23_FRB'] = self.connect_exc(
            populations['L4_E'], populations['L23_E_FRB'])
        connections['L4_to_L5_RS'] = self.connect_exc(
            populations['L4_E'], populations['L5_E_RS'])
        connections['L4_to_L5_IB'] = self.connect_exc(
            populations['L4_E'], populations['L5_E_IB'])
        connections['L4_to_L4_I'] = self.connect_exc(
            populations['L4_E'], populations['L4_I'])
        
        # L2/3 excitatory connections
        connections['L23_RS_to_RS'] = self.connect_exc(
            populations['L23_E_RS'], populations['L23_E_RS'])
        connections['L23_FRB_to_FRB'] = self.connect_exc(
            populations['L23_E_FRB'], populations['L23_E_FRB'])
        connections['L23_RS_to_FRB'] = self.connect_exc(
            populations['L23_E_RS'], populations['L23_E_FRB'])
        connections['L23_FRB_to_RS'] = self.connect_exc(
            populations['L23_E_FRB'], populations['L23_E_RS'])
        connections['L23_RS_to_L5_RS'] = self.connect_exc(
            populations['L23_E_RS'], populations['L5_E_RS'])
        connections['L23_RS_to_L5_IB'] = self.connect_exc(
            populations['L23_E_RS'], populations['L5_E_IB'])
        connections['L23_RS_to_L6'] = self.connect_exc(
            populations['L23_E_RS'], populations['L6_E'])
        connections['L23_FRB_to_L5_RS'] = self.connect_exc(
            populations['L23_E_FRB'], populations['L5_E_RS'])
        connections['L23_FRB_to_L5_IB'] = self.connect_exc(
            populations['L23_E_FRB'], populations['L5_E_IB'])
        connections['L23_FRB_to_L6'] = self.connect_exc(
            populations['L23_E_FRB'], populations['L6_E'])
        
        # L5 connections
        connections['L5_RS_to_RS'] = self.connect_exc(
            populations['L5_E_RS'], populations['L5_E_RS'])
        connections['L5_IB_to_IB'] = self.connect_exc(
            populations['L5_E_IB'], populations['L5_E_IB'])
        connections['L5_RS_to_IB'] = self.connect_exc(
            populations['L5_E_RS'], populations['L5_E_IB'])
        connections['L5_IB_to_RS'] = self.connect_exc(
            populations['L5_E_IB'], populations['L5_E_RS'])
        connections['L5_RS_to_L6'] = self.connect_exc(
            populations['L5_E_RS'], populations['L6_E'])
        connections['L5_IB_to_L6'] = self.connect_exc(
            populations['L5_E_IB'], populations['L6_E'])
        
        # L6 connections
        connections['L6_to_L5_RS'] = self.connect_exc(
            populations['L6_E'], populations['L5_E_RS'])
        connections['L6_to_L5_IB'] = self.connect_exc(
            populations['L6_E'], populations['L5_E_IB'])
        
        # Inhibitory connections
        connections['L4_I_to_L4'] = self.connect_inh(
            populations['L4_I'], populations['L4_E'])
        connections['L23_LTS_to_RS'] = self.connect_inh(
            populations['L23_I_LTS'], populations['L23_E_RS'])
        connections['L23_LTS_to_FRB'] = self.connect_inh(
            populations['L23_I_LTS'], populations['L23_E_FRB'])
        connections['L23_Bask_to_RS'] = self.connect_inh(
            populations['L23_I_Bask'], populations['L23_E_RS'])
        connections['L23_Bask_to_FRB'] = self.connect_inh(
            populations['L23_I_Bask'], populations['L23_E_FRB'])
        connections['L56_LTS_to_L5_RS'] = self.connect_inh(
            populations['L56_I_LTS'], populations['L5_E_RS'])
        connections['L56_LTS_to_L5_IB'] = self.connect_inh(
            populations['L56_I_LTS'], populations['L5_E_IB'])
        connections['L56_Bask_to_L5_RS'] = self.connect_inh(
            populations['L56_I_Bask'], populations['L5_E_RS'])
        connections['L56_Bask_to_L6'] = self.connect_inh(
            populations['L56_I_Bask'], populations['L6_E'])
        
        return connections
    
    def run(self) -> Dict[str, np.ndarray]:
        """Run the full simulation and return spike data."""
        if not self._neuron_imported:
            return self._mock_run()
        
        h = self.h
        
        # Build network
        populations = self.build_network()
        connections = self.connect_network(populations)
        
        # Apply stimulus to thalamus
        stim_syn, stim_nc, stim_ns = self.stimulate_group(populations['thalamus_E'])
        
        # Run simulation
        h.tstop = self.params.tstop
        h.finitialize(self.network_config.v_init)
        h.continuerun(h.tstop)
        
        # Extract spike times
        spike_data = {}
        
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': (self.flatten(populations['L23_E_RS']) + 
                   self.flatten(populations['L23_E_FRB'])),
            'L5': (self.flatten(populations['L5_E_RS']) + 
                  self.flatten(populations['L5_E_IB'])),
            'L6': self.flatten(populations['L6_E']),
        }
        
        for layer_name, neurons in layer_groups.items():
            all_spikes = []
            for neuron in neurons:
                spikes = self._extract_spike_times(neuron)
                all_spikes.extend(spikes)
            spike_data[layer_name] = np.array(sorted(all_spikes))
        
        return spike_data
    
    def _extract_spike_times(self, neuron, threshold: float = 0, 
                            refractory_period: float = 2.0) -> List[float]:
        """Extract spike times from voltage trace (from notebook)."""
        v = np.array(neuron.vvec)
        t = np.array(neuron.tvec)
        
        spike_times = []
        last_spike_time = -np.inf
        
        for i in range(1, len(v)):
            if v[i-1] < threshold <= v[i]:
                if (t[i] - last_spike_time) >= refractory_period:
                    spike_times.append(t[i])
                    last_spike_time = t[i]
        
        return spike_times
    
    def _mock_run(self) -> Dict[str, np.ndarray]:
        """Generate mock spike data when NEURON is not available."""
        np.random.seed(42)
        p = self.params
        cfg = self.network_config
        
        # Simulate expected activation sequence with noise
        base_latencies = {
            'thalamus': 5,
            'L4': 12,
            'L23': 20,
            'L5': 28,
            'L6': 35,
        }
        
        # Scale spike counts by population size
        pop_scales = {
            'thalamus': cfg.N_thalamus_E / 5,
            'L4': cfg.N_L4_E / 24,
            'L23': cfg.N_L23_E / 24,
            'L5': cfg.N_L5_E / 20,
            'L6': cfg.N_L6_E / 16,
        }
        
        spike_data = {}
        for layer, base_lat in base_latencies.items():
            base_n = int(np.random.poisson(15))
            n_spikes = int(base_n * pop_scales.get(layer, 1.0))
            
            latency_shift = (p.exc_weight_mean - 0.001) * 5000
            spikes = np.random.exponential(10, n_spikes) + base_lat + latency_shift
            spikes = spikes[spikes < p.tstop]
            spike_data[layer] = np.sort(spikes)
        
        return spike_data


class MockNeuron:
    """Mock neuron for testing without NEURON installed."""
    def __init__(self, inh=False):
        self.inh = inh
        self.vvec = []
        self.tvec = []


def load_config(path: Optional[Union[str, Path]] = None) -> NetworkConfig:
    """Load network configuration from file or return defaults."""
    if path is None:
        # Try default locations
        default_paths = [
            Path(__file__).parent.parent / 'config' / 'network.yaml',
            Path(__file__).parent.parent / 'config' / 'network.json',
            Path('config/network.yaml'),
            Path('config/network.json'),
        ]
        for p in default_paths:
            if p.exists():
                return NetworkConfig.from_file(p)
        return NetworkConfig()
    return NetworkConfig.from_file(path)


def run_simulation(params_dict: dict, 
                   network_config: Optional[NetworkConfig] = None,
                   config_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Run simulation with given parameters and return spike data."""
    params = SimulationParams.from_dict(params_dict)
    sim = ThalamoCorticalSimulator(
        params=params, 
        network_config=network_config,
        config_path=config_path
    )
    return sim.run()