"""Simulator module wrapping HHNeuron thalamo-cortical column model."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import copy


@dataclass
class SimulationConfig:
    """Configuration for simulation execution (CPU/GPU settings)."""
    use_gpu: bool = False
    gpu_device: int = 0
    coreneuron_datadir: str = "./coreneuron_data"
    num_threads: int = 1
    use_mpi: bool = False
    cache_efficient: bool = True
    record_spikes: bool = True
    spike_output_file: str = "out.dat"
    verbose: bool = False
    
    def setup_gpu_environment(self):
        if self.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_device)
            if self.verbose:
                print(f"Set CUDA_VISIBLE_DEVICES={self.gpu_device}")
    
    def get_coreneuron_datadir_arg(self) -> str:
        return f"-d {self.coreneuron_datadir}"


DEFAULT_NETWORK_CONFIG = {
    'network': {
        'thalamus': {'E': 5, 'I': 1},
        'L4': {'E': 24, 'I': 6},
        'L23': {'E': 24, 'I': 6},
        'L5': {'E': 20, 'I': 5},
        'L6': {'E': 16, 'I': 4},
    },
    'splits': {
        'L23_E': 2, 'L23_I': 3, 'L4_E': 1, 'L4_I': 1,
        'L5_E': 2, 'L56_I': 3, 'L6_E': 1,
    },
    'simulation': {'tstop': 200.0, 'dt': 0.025, 'v_init': -65}
}


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""
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
    split_L23_E: int = 2
    split_L23_I: int = 3
    split_L4_E: int = 1
    split_L4_I: int = 1
    split_L5_E: int = 2
    split_L56_I: int = 3
    split_L6_E: int = 1
    tstop: float = 200.0
    dt: float = 0.025
    v_init: float = -65.0
    
    @classmethod
    def from_dict(cls, config: dict) -> 'NetworkConfig':
        kwargs = {}
        if 'network' in config:
            net = config['network']
            for layer, prefix in [('thalamus', 'N_thalamus'), ('L4', 'N_L4'), 
                                  ('L23', 'N_L23'), ('L5', 'N_L5'), ('L6', 'N_L6')]:
                if layer in net:
                    kwargs[f'{prefix}_E'] = net[layer].get('E', kwargs.get(f'{prefix}_E', 5))
                    kwargs[f'{prefix}_I'] = net[layer].get('I', kwargs.get(f'{prefix}_I', 1))
        if 'splits' in config:
            splits = config['splits']
            for key in ['L23_E', 'L23_I', 'L4_E', 'L4_I', 'L5_E', 'L56_I', 'L6_E']:
                if key in splits:
                    kwargs[f'split_{key}'] = splits[key]
        if 'simulation' in config:
            sim = config['simulation']
            kwargs['tstop'] = sim.get('tstop', 200.0)
            kwargs['dt'] = sim.get('dt', 0.025)
            kwargs['v_init'] = sim.get('v_init', -65.0)
        return cls(**kwargs)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'NetworkConfig':
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
        return {
            'network': {
                'thalamus': {'E': self.N_thalamus_E, 'I': self.N_thalamus_I},
                'L4': {'E': self.N_L4_E, 'I': self.N_L4_I},
                'L23': {'E': self.N_L23_E, 'I': self.N_L23_I},
                'L5': {'E': self.N_L5_E, 'I': self.N_L5_I},
                'L6': {'E': self.N_L6_E, 'I': self.N_L6_I},
            },
            'splits': {
                'L23_E': self.split_L23_E, 'L23_I': self.split_L23_I,
                'L4_E': self.split_L4_E, 'L4_I': self.split_L4_I,
                'L5_E': self.split_L5_E, 'L56_I': self.split_L56_I, 'L6_E': self.split_L6_E,
            },
            'simulation': {'tstop': self.tstop, 'dt': self.dt, 'v_init': self.v_init}
        }
    
    def save(self, path: Union[str, Path]):
        path = Path(path)
        config = self.to_dict()
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    yaml.dump(config, f, default_flow_style=False)
                except ImportError:
                    json.dump(config, f, indent=2)
            else:
                json.dump(config, f, indent=2)
    
    @property
    def total_neurons(self) -> int:
        return (self.N_thalamus_E + self.N_thalamus_I + self.N_L4_E + self.N_L4_I +
                self.N_L23_E + self.N_L23_I + self.N_L5_E + self.N_L5_I + self.N_L6_E + self.N_L6_I)
    
    def summary(self) -> str:
        return '\n'.join([
            "Network Configuration:",
            f"  Thalamus: {self.N_thalamus_E}E / {self.N_thalamus_I}I",
            f"  L4:       {self.N_L4_E}E / {self.N_L4_I}I",
            f"  L2/3:     {self.N_L23_E}E / {self.N_L23_I}I",
            f"  L5:       {self.N_L5_E}E / {self.N_L5_I}I",
            f"  L6:       {self.N_L6_E}E / {self.N_L6_I}I",
            f"  Total:    {self.total_neurons} neurons",
        ])


@dataclass
class SimulationParams:
    """Parameters for synaptic connections and stimulation."""
    exc_weight_mean: float = 0.001
    exc_weight_std: float = 0.0009
    exc_tau: float = 2.0
    exc_delay_mean: float = 3.0
    exc_delay_std: float = 2.0
    exc_e: float = 0.0
    min_delay: float = 0.2
    inh_weight: float = 0.001
    inh_tau: float = 3.0
    inh_delay: float = 2.0
    inh_e: float = -75.0
    conn_prob: float = 0.1
    stim_start: float = 0.0
    stim_number: int = 20
    stim_interval: float = 15.0
    stim_noise: float = 1.0
    stim_weight: float = 0.01
    tstop: float = 200.0
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SimulationParams':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ThalamoCorticalSimulator:
    """Simulates a thalamo-cortical column with HH neurons."""
    
    _neuron_warning_shown = False
    _coreneuron_warning_shown = False
    
    def __init__(self, params: Optional[SimulationParams] = None,
                 network_config: Optional[NetworkConfig] = None,
                 config_path: Optional[Union[str, Path]] = None,
                 sim_config: Optional[SimulationConfig] = None):
        self.params = params or SimulationParams()
        self.sim_config = sim_config or SimulationConfig()
        
        if network_config is not None:
            self.network_config = network_config
        elif config_path is not None:
            self.network_config = NetworkConfig.from_file(config_path)
        else:
            self.network_config = NetworkConfig()
        
        if self.params.tstop == 200.0:
            self.params.tstop = self.network_config.tstop
        
        self._neuron_imported = False
        self._coreneuron_available = False
        self.pc = None
        self._setup_neuron()
    
    def _setup_neuron(self):
        if self.sim_config.use_gpu:
            self.sim_config.setup_gpu_environment()
        
        try:
            from neuron import h
            h.load_file('stdrun.hoc')
            self.h = h
            self._neuron_imported = True
            self._use_gpu_override = None
            self.pc = h.ParallelContext()
            
            if self.sim_config.use_gpu:
                self._setup_coreneuron()
            else:
                self._setup_cpu_threading()
        except ImportError:
            if not ThalamoCorticalSimulator._neuron_warning_shown:
                print("Note: NEURON not available. Running in mock mode.")
                ThalamoCorticalSimulator._neuron_warning_shown = True
            self.h = None
            self._neuron_imported = False
            self._use_gpu_override = None
    
    def _setup_cpu_threading(self):
        h = self.h
        num_threads = self.sim_config.num_threads
        
        if self.sim_config.use_mpi:
            num_threads = min(num_threads, 2)
        else:
            MAX_SAFE_THREADS = 8
            if num_threads > MAX_SAFE_THREADS:
                if self.sim_config.verbose:
                    print(f"WARNING: Capping threads from {num_threads} to {MAX_SAFE_THREADS}")
                num_threads = MAX_SAFE_THREADS
        
        if num_threads > 1:
            self.pc.nthread(num_threads, 1)
        
        if self.sim_config.cache_efficient:
            self.pc.optimize_node_order(1)
        
        if self.sim_config.use_mpi and self.sim_config.verbose:
            rank = int(self.pc.id())
            nhost = int(self.pc.nhost())
            if rank == 0:
                print(f"MPI enabled: {nhost} ranks")
    
    def _setup_coreneuron(self):
        try:
            from neuron import coreneuron
            coreneuron.enable = True
            coreneuron.gpu = True
            
            datadir = Path(self.sim_config.coreneuron_datadir)
            datadir.mkdir(parents=True, exist_ok=True)
            self._coreneuron_available = True
            
            if self.sim_config.verbose:
                cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
                print(f"CoreNEURON GPU mode enabled (CUDA_VISIBLE_DEVICES={cuda_device})")
        except ImportError:
            if not ThalamoCorticalSimulator._coreneuron_warning_shown:
                print("Warning: CoreNEURON not available. Falling back to CPU mode.")
                ThalamoCorticalSimulator._coreneuron_warning_shown = True
            self._coreneuron_available = False
            self._use_gpu_override = False
            self._setup_cpu_threading()
        except Exception as e:
            print(f"Warning: CoreNEURON setup failed: {e}. Falling back to CPU mode.")
            self._coreneuron_available = False
            self._use_gpu_override = False
            self._setup_cpu_threading()
    
    def _create_hh_neuron(self, inh: bool = False, record_voltage: bool = False):
        if not self._neuron_imported:
            return MockNeuron(inh)
        
        h = self.h
        record = record_voltage
        
        class HHNeuron:
            def __init__(self, inh=False):
                self.soma = h.Section(name='soma')
                self.soma.L = 20
                self.soma.diam = 20
                self.soma.insert('hh')
                self.inh = inh
                
                if record:
                    self.vvec = h.Vector()
                    self.vvec.record(self.soma(0.5)._ref_v)
                    self.tvec = h.Vector()
                    self.tvec.record(h._ref_t)
                else:
                    self.vvec = None
                    self.tvec = None
        
        return HHNeuron(inh)
    
    @staticmethod
    def flatten(population):
        if isinstance(population, list) and len(population) > 0 and isinstance(population[0], list):
            return [neuron for subgroup in population for neuron in subgroup]
        return population
    
    @staticmethod
    def split_population(population, n_subgroups):
        if n_subgroups <= 0:
            return [population]
        size = len(population)
        if size == 0:
            return [[] for _ in range(n_subgroups)]
        step = max(1, size // n_subgroups)
        return [population[i*step:(i+1)*step] for i in range(n_subgroups)]
    
    def connect_exc(self, source_neurons, target_neurons) -> Tuple[list, list]:
        import random
        source_neurons = self.flatten(source_neurons)
        target_neurons = self.flatten(target_neurons)
        netcons, synapses = [], []
        
        if not self._neuron_imported:
            return synapses, netcons
        
        h, p = self.h, self.params
        
        for src in source_neurons:
            for tgt in target_neurons:
                if p.conn_prob < 1.0 and random.random() > p.conn_prob:
                    continue
                syn = h.ExpSyn(tgt.soma(0.5))
                syn.e = p.exc_e
                syn.tau = p.exc_tau
                nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
                nc.threshold = 0
                nc.weight[0] = max(0.0, random.gauss(p.exc_weight_mean, p.exc_weight_std))
                nc.delay = max(p.min_delay, random.gauss(p.exc_delay_mean, p.exc_delay_std))
                synapses.append(syn)
                netcons.append(nc)
        return synapses, netcons
    
    def connect_inh(self, source_neurons, target_neurons) -> Tuple[list, list]:
        import random
        source_neurons = self.flatten(source_neurons)
        target_neurons = self.flatten(target_neurons)
        netcons, synapses = [], []
        
        if not self._neuron_imported:
            return synapses, netcons
        
        h, p = self.h, self.params
        
        for src in source_neurons:
            for tgt in target_neurons:
                if p.conn_prob < 1.0 and random.random() > p.conn_prob:
                    continue
                syn = h.ExpSyn(tgt.soma(0.5))
                syn.e = p.inh_e
                syn.tau = p.inh_tau
                nc = h.NetCon(src.soma(0.5)._ref_v, syn, sec=src.soma)
                nc.threshold = 0
                nc.weight[0] = p.inh_weight
                nc.delay = max(p.min_delay, p.inh_delay)
                synapses.append(syn)
                netcons.append(nc)
        return synapses, netcons
    
    def stimulate_group(self, group) -> Tuple[list, list, list]:
        syn_inputs, conns, netstims = [], [], []
        if not self._neuron_imported:
            return syn_inputs, conns, netstims
        
        h, p = self.h, self.params
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
        cfg = self.network_config
        return {
            'thalamus_E': [self._create_hh_neuron(inh=False) for _ in range(cfg.N_thalamus_E)],
            'thalamus_I': [self._create_hh_neuron(inh=True) for _ in range(cfg.N_thalamus_I)],
            'L4_E': self.split_population([self._create_hh_neuron() for _ in range(cfg.N_L4_E)], cfg.split_L4_E),
            'L4_I': self.split_population([self._create_hh_neuron(inh=True) for _ in range(cfg.N_L4_I)], cfg.split_L4_I),
            'L23_E_RS': self.split_population([self._create_hh_neuron() for _ in range(cfg.N_L23_E)], cfg.split_L23_E),
            'L23_E_FRB': self.split_population([self._create_hh_neuron() for _ in range(cfg.N_L23_E)], cfg.split_L23_E),
            'L23_I_Bask': self.split_population([self._create_hh_neuron(inh=True) for _ in range(cfg.N_L23_I)], cfg.split_L23_I),
            'L23_I_LTS': self.split_population([self._create_hh_neuron(inh=True) for _ in range(cfg.N_L23_I)], cfg.split_L23_I),
            'L23_I_Axax': self.split_population([self._create_hh_neuron(inh=True) for _ in range(cfg.N_L23_I)], cfg.split_L23_I),
            'L5_E_RS': self.split_population([self._create_hh_neuron() for _ in range(cfg.N_L5_E)], cfg.split_L5_E),
            'L5_E_IB': self.split_population([self._create_hh_neuron() for _ in range(cfg.N_L5_E)], cfg.split_L5_E),
            'L56_I_Bask': self.split_population([self._create_hh_neuron(inh=True) for _ in range(cfg.N_L5_I + cfg.N_L6_I)], cfg.split_L56_I),
            'L56_I_LTS': self.split_population([self._create_hh_neuron(inh=True) for _ in range(cfg.N_L5_I + cfg.N_L6_I)], cfg.split_L56_I),
            'L56_I_Axax': self.split_population([self._create_hh_neuron(inh=True) for _ in range(cfg.N_L5_I + cfg.N_L6_I)], cfg.split_L56_I),
            'L6_E': self.split_population([self._create_hh_neuron() for _ in range(cfg.N_L6_E)], cfg.split_L6_E),
        }
    
    def connect_network(self, populations: Dict[str, list]) -> Dict[str, Tuple[list, list]]:
        connections = {}
        connections['TCR_to_L4'] = self.connect_exc(populations['thalamus_E'], populations['L4_E'])
        connections['TCR_to_nRT'] = self.connect_exc(populations['thalamus_E'], populations['thalamus_I'])
        connections['L4_to_L4'] = self.connect_exc(populations['L4_E'], populations['L4_E'])
        connections['L4_to_L23_RS'] = self.connect_exc(populations['L4_E'], populations['L23_E_RS'])
        connections['L4_to_L23_FRB'] = self.connect_exc(populations['L4_E'], populations['L23_E_FRB'])
        connections['L4_to_L5_RS'] = self.connect_exc(populations['L4_E'], populations['L5_E_RS'])
        connections['L4_to_L5_IB'] = self.connect_exc(populations['L4_E'], populations['L5_E_IB'])
        connections['L4_to_L4_I'] = self.connect_exc(populations['L4_E'], populations['L4_I'])
        connections['L23_RS_to_RS'] = self.connect_exc(populations['L23_E_RS'], populations['L23_E_RS'])
        connections['L23_FRB_to_FRB'] = self.connect_exc(populations['L23_E_FRB'], populations['L23_E_FRB'])
        connections['L23_RS_to_FRB'] = self.connect_exc(populations['L23_E_RS'], populations['L23_E_FRB'])
        connections['L23_FRB_to_RS'] = self.connect_exc(populations['L23_E_FRB'], populations['L23_E_RS'])
        connections['L23_RS_to_L5_RS'] = self.connect_exc(populations['L23_E_RS'], populations['L5_E_RS'])
        connections['L23_RS_to_L5_IB'] = self.connect_exc(populations['L23_E_RS'], populations['L5_E_IB'])
        connections['L23_RS_to_L6'] = self.connect_exc(populations['L23_E_RS'], populations['L6_E'])
        connections['L23_FRB_to_L5_RS'] = self.connect_exc(populations['L23_E_FRB'], populations['L5_E_RS'])
        connections['L23_FRB_to_L5_IB'] = self.connect_exc(populations['L23_E_FRB'], populations['L5_E_IB'])
        connections['L23_FRB_to_L6'] = self.connect_exc(populations['L23_E_FRB'], populations['L6_E'])
        connections['L5_RS_to_RS'] = self.connect_exc(populations['L5_E_RS'], populations['L5_E_RS'])
        connections['L5_IB_to_IB'] = self.connect_exc(populations['L5_E_IB'], populations['L5_E_IB'])
        connections['L5_RS_to_IB'] = self.connect_exc(populations['L5_E_RS'], populations['L5_E_IB'])
        connections['L5_IB_to_RS'] = self.connect_exc(populations['L5_E_IB'], populations['L5_E_RS'])
        connections['L5_RS_to_L6'] = self.connect_exc(populations['L5_E_RS'], populations['L6_E'])
        connections['L5_IB_to_L6'] = self.connect_exc(populations['L5_E_IB'], populations['L6_E'])
        connections['L6_to_L5_RS'] = self.connect_exc(populations['L6_E'], populations['L5_E_RS'])
        connections['L6_to_L5_IB'] = self.connect_exc(populations['L6_E'], populations['L5_E_IB'])
        connections['L4_I_to_L4'] = self.connect_inh(populations['L4_I'], populations['L4_E'])
        connections['L23_LTS_to_RS'] = self.connect_inh(populations['L23_I_LTS'], populations['L23_E_RS'])
        connections['L23_LTS_to_FRB'] = self.connect_inh(populations['L23_I_LTS'], populations['L23_E_FRB'])
        connections['L23_Bask_to_RS'] = self.connect_inh(populations['L23_I_Bask'], populations['L23_E_RS'])
        connections['L23_Bask_to_FRB'] = self.connect_inh(populations['L23_I_Bask'], populations['L23_E_FRB'])
        connections['L56_LTS_to_L5_RS'] = self.connect_inh(populations['L56_I_LTS'], populations['L5_E_RS'])
        connections['L56_LTS_to_L5_IB'] = self.connect_inh(populations['L56_I_LTS'], populations['L5_E_IB'])
        connections['L56_Bask_to_L5_RS'] = self.connect_inh(populations['L56_I_Bask'], populations['L5_E_RS'])
        connections['L56_Bask_to_L6'] = self.connect_inh(populations['L56_I_Bask'], populations['L6_E'])
        return connections
    
    def _should_use_gpu(self) -> bool:
        if self._use_gpu_override is not None:
            return self._use_gpu_override
        return self.sim_config.use_gpu and self._coreneuron_available
    
    def run(self, return_traces: bool = False) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Optional[Dict]]]:
        if not self._neuron_imported:
            spike_data = self._mock_run()
            return (spike_data, None) if return_traces else spike_data
        
        if self._should_use_gpu():
            spike_data, trace_data = self._run_coreneuron(return_traces=return_traces)
        elif self.sim_config.use_mpi:
            spike_data, trace_data = self._run_mpi(return_traces=return_traces)
        else:
            spike_data, trace_data = self._run_standard(return_traces=return_traces)
        
        return (spike_data, trace_data) if return_traces else spike_data
    
    def _run_standard(self, return_traces: bool = False) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
        h = self.h
        populations = self.build_network()
        
        if return_traces:
            self._enable_voltage_recording(populations)
        
        connections = self.connect_network(populations)
        stim_syn, stim_nc, stim_ns = self.stimulate_group(populations['thalamus_E'])
        spike_recorders = self._setup_spike_recorders(populations)
        
        h.tstop = self.params.tstop
        h.dt = self.network_config.dt
        h.finitialize(self.network_config.v_init)
        h.continuerun(h.tstop)
        
        spike_data = self._collect_spikes(spike_recorders)
        trace_data = self._extract_traces(populations) if return_traces else None
        return spike_data, trace_data
    
    def _enable_voltage_recording(self, populations: Dict[str, list]):
        h = self.h
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': self.flatten(populations['L23_E_RS']) + self.flatten(populations['L23_E_FRB']),
            'L5': self.flatten(populations['L5_E_RS']) + self.flatten(populations['L5_E_IB']),
            'L6': self.flatten(populations['L6_E']),
        }
        for layer_name, neurons in layer_groups.items():
            if neurons and len(neurons) > 0:
                neuron = neurons[0]
                if neuron.vvec is None:
                    neuron.vvec = h.Vector()
                    neuron.vvec.record(neuron.soma(0.5)._ref_v)
                    neuron.tvec = h.Vector()
                    neuron.tvec.record(h._ref_t)
    
    def _run_mpi(self, return_traces: bool = False) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
        h, pc = self.h, self.pc
        rank, nhost = int(pc.id()), int(pc.nhost())
        
        populations = self.build_network()
        if return_traces and rank == 0:
            self._enable_voltage_recording(populations)
        
        gid_info = self._assign_gids_mpi(populations, rank, nhost)
        connections = self._connect_network_mpi(populations, gid_info, rank, nhost)
        
        local_thalamus = [n for i, n in enumerate(populations['thalamus_E']) if i % nhost == rank]
        if local_thalamus:
            self.stimulate_group(local_thalamus)
        
        spike_times, spike_gids = h.Vector(), h.Vector()
        pc.spike_record(-1, spike_times, spike_gids)
        pc.set_maxstep(10)
        
        h.tstop = self.params.tstop
        h.dt = self.network_config.dt
        h.finitialize(self.network_config.v_init)
        pc.psolve(h.tstop)
        
        spike_data = self._gather_spikes_mpi(spike_times, spike_gids, gid_info)
        trace_data = self._extract_traces(populations) if return_traces and rank == 0 else None
        return spike_data, trace_data
    
    def _assign_gids_mpi(self, populations: Dict[str, list], rank: int, nhost: int) -> Dict:
        h, pc = self.h, self.pc
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': self.flatten(populations['L23_E_RS']) + self.flatten(populations['L23_E_FRB']),
            'L5': self.flatten(populations['L5_E_RS']) + self.flatten(populations['L5_E_IB']),
            'L6': self.flatten(populations['L6_E']),
        }
        
        gid_info = {'gid_to_layer': {}, 'layer_gids': {}, 'local_gids': set()}
        gid = 0
        
        for layer_name, neurons in layer_groups.items():
            gid_info['layer_gids'][layer_name] = []
            for i, neuron in enumerate(neurons):
                owner_rank = gid % nhost
                gid_info['gid_to_layer'][gid] = layer_name
                gid_info['layer_gids'][layer_name].append(gid)
                if owner_rank == rank:
                    gid_info['local_gids'].add(gid)
                    pc.set_gid2node(gid, rank)
                    nc = h.NetCon(neuron.soma(0.5)._ref_v, None, sec=neuron.soma)
                    nc.threshold = 0
                    pc.cell(gid, nc)
                gid += 1
        return gid_info
    
    def _connect_network_mpi(self, populations: Dict[str, list], gid_info: Dict, 
                             rank: int, nhost: int) -> Dict:
        import random
        h, pc, p = self.h, self.pc, self.params
        
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': self.flatten(populations['L23_E_RS']) + self.flatten(populations['L23_E_FRB']),
            'L5': self.flatten(populations['L5_E_RS']) + self.flatten(populations['L5_E_IB']),
            'L6': self.flatten(populations['L6_E']),
        }
        
        neuron_to_gid = {}
        gid = 0
        for layer_name, neurons in layer_groups.items():
            for neuron in neurons:
                neuron_to_gid[id(neuron)] = gid
                gid += 1
        
        def connect_mpi(source_neurons, target_neurons, exc=True):
            src_list, tgt_list = self.flatten(source_neurons), self.flatten(target_neurons)
            netcons, synapses = [], []
            for src in src_list:
                src_gid = neuron_to_gid[id(src)]
                for tgt in tgt_list:
                    tgt_gid = neuron_to_gid[id(tgt)]
                    if tgt_gid not in gid_info['local_gids']:
                        continue
                    if p.conn_prob < 1.0 and random.random() > p.conn_prob:
                        continue
                    syn = h.ExpSyn(tgt.soma(0.5))
                    if exc:
                        syn.e, syn.tau = p.exc_e, p.exc_tau
                        w = max(0.0, random.gauss(p.exc_weight_mean, p.exc_weight_std))
                        d = max(p.min_delay, random.gauss(p.exc_delay_mean, p.exc_delay_std))
                    else:
                        syn.e, syn.tau = p.inh_e, p.inh_tau
                        w, d = p.inh_weight, max(p.min_delay, p.inh_delay)
                    nc = pc.gid_connect(src_gid, syn)
                    nc.weight[0], nc.delay = w, d
                    synapses.append(syn)
                    netcons.append(nc)
            return synapses, netcons
        
        connections = {}
        connections['TCR_to_L4'] = connect_mpi(populations['thalamus_E'], populations['L4_E'])
        connections['TCR_to_nRT'] = connect_mpi(populations['thalamus_E'], populations['thalamus_I'])
        connections['L4_to_L4'] = connect_mpi(populations['L4_E'], populations['L4_E'])
        connections['L4_to_L23_RS'] = connect_mpi(populations['L4_E'], populations['L23_E_RS'])
        connections['L4_to_L23_FRB'] = connect_mpi(populations['L4_E'], populations['L23_E_FRB'])
        connections['L4_to_L5_RS'] = connect_mpi(populations['L4_E'], populations['L5_E_RS'])
        connections['L4_to_L5_IB'] = connect_mpi(populations['L4_E'], populations['L5_E_IB'])
        connections['L4_to_L4_I'] = connect_mpi(populations['L4_E'], populations['L4_I'])
        connections['L23_RS_to_RS'] = connect_mpi(populations['L23_E_RS'], populations['L23_E_RS'])
        connections['L23_FRB_to_FRB'] = connect_mpi(populations['L23_E_FRB'], populations['L23_E_FRB'])
        connections['L23_RS_to_FRB'] = connect_mpi(populations['L23_E_RS'], populations['L23_E_FRB'])
        connections['L23_FRB_to_RS'] = connect_mpi(populations['L23_E_FRB'], populations['L23_E_RS'])
        connections['L23_RS_to_L5_RS'] = connect_mpi(populations['L23_E_RS'], populations['L5_E_RS'])
        connections['L23_RS_to_L5_IB'] = connect_mpi(populations['L23_E_RS'], populations['L5_E_IB'])
        connections['L23_RS_to_L6'] = connect_mpi(populations['L23_E_RS'], populations['L6_E'])
        connections['L23_FRB_to_L5_RS'] = connect_mpi(populations['L23_E_FRB'], populations['L5_E_RS'])
        connections['L23_FRB_to_L5_IB'] = connect_mpi(populations['L23_E_FRB'], populations['L5_E_IB'])
        connections['L23_FRB_to_L6'] = connect_mpi(populations['L23_E_FRB'], populations['L6_E'])
        connections['L5_RS_to_RS'] = connect_mpi(populations['L5_E_RS'], populations['L5_E_RS'])
        connections['L5_IB_to_IB'] = connect_mpi(populations['L5_E_IB'], populations['L5_E_IB'])
        connections['L5_RS_to_IB'] = connect_mpi(populations['L5_E_RS'], populations['L5_E_IB'])
        connections['L5_IB_to_RS'] = connect_mpi(populations['L5_E_IB'], populations['L5_E_RS'])
        connections['L5_RS_to_L6'] = connect_mpi(populations['L5_E_RS'], populations['L6_E'])
        connections['L5_IB_to_L6'] = connect_mpi(populations['L5_E_IB'], populations['L6_E'])
        connections['L6_to_L5_RS'] = connect_mpi(populations['L6_E'], populations['L5_E_RS'])
        connections['L6_to_L5_IB'] = connect_mpi(populations['L6_E'], populations['L5_E_IB'])
        connections['L4_I_to_L4'] = connect_mpi(populations['L4_I'], populations['L4_E'], exc=False)
        connections['L23_LTS_to_RS'] = connect_mpi(populations['L23_I_LTS'], populations['L23_E_RS'], exc=False)
        connections['L23_LTS_to_FRB'] = connect_mpi(populations['L23_I_LTS'], populations['L23_E_FRB'], exc=False)
        connections['L23_Bask_to_RS'] = connect_mpi(populations['L23_I_Bask'], populations['L23_E_RS'], exc=False)
        connections['L23_Bask_to_FRB'] = connect_mpi(populations['L23_I_Bask'], populations['L23_E_FRB'], exc=False)
        connections['L56_LTS_to_L5_RS'] = connect_mpi(populations['L56_I_LTS'], populations['L5_E_RS'], exc=False)
        connections['L56_LTS_to_L5_IB'] = connect_mpi(populations['L56_I_LTS'], populations['L5_E_IB'], exc=False)
        connections['L56_Bask_to_L5_RS'] = connect_mpi(populations['L56_I_Bask'], populations['L5_E_RS'], exc=False)
        connections['L56_Bask_to_L6'] = connect_mpi(populations['L56_I_Bask'], populations['L6_E'], exc=False)
        return connections
    
    def _gather_spikes_mpi(self, spike_times, spike_gids, gid_info) -> Dict[str, np.ndarray]:
        local_times = np.array(spike_times)
        local_gids = np.array(spike_gids, dtype=int)
        
        spike_data = {layer: [] for layer in gid_info['layer_gids'].keys()}
        for t, gid in zip(local_times, local_gids):
            if gid in gid_info['gid_to_layer']:
                spike_data[gid_info['gid_to_layer'][gid]].append(t)
        
        for layer in spike_data:
            spike_data[layer] = np.sort(np.array(spike_data[layer]))
        return spike_data
    
    def _setup_spike_recorders(self, populations: Dict[str, list]) -> Dict[str, Tuple]:
        h = self.h
        recorders = {}
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': self.flatten(populations['L23_E_RS']) + self.flatten(populations['L23_E_FRB']),
            'L5': self.flatten(populations['L5_E_RS']) + self.flatten(populations['L5_E_IB']),
            'L6': self.flatten(populations['L6_E']),
        }
        
        for layer_name, neurons in layer_groups.items():
            spike_times = h.Vector()
            netcons = []
            for neuron in neurons:
                nc = h.NetCon(neuron.soma(0.5)._ref_v, None, sec=neuron.soma)
                nc.threshold = 0
                nc.record(spike_times)
                netcons.append(nc)
            recorders[layer_name] = (spike_times, netcons)
        return recorders
    
    def _collect_spikes(self, recorders: Dict[str, Tuple]) -> Dict[str, np.ndarray]:
        return {layer_name: np.sort(np.array(spike_times)) 
                for layer_name, (spike_times, netcons) in recorders.items()}
    
    def _run_coreneuron(self, return_traces: bool = False) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
        h = self.h
        populations = self.build_network()
        connections = self.connect_network(populations)
        stim_syn, stim_nc, stim_ns = self.stimulate_group(populations['thalamus_E'])
        self._setup_spike_recording(populations)
        
        h.tstop = self.params.tstop
        h.finitialize(self.network_config.v_init)
        
        datadir_arg = self.sim_config.get_coreneuron_datadir_arg()
        if self.sim_config.verbose:
            print(f"Running CoreNEURON (GPU={self._coreneuron_available})")
        
        self.pc.nrncore_run(datadir_arg)
        spike_data = self._parse_coreneuron_spikes(populations)
        
        trace_data = None
        if return_traces and self.sim_config.verbose:
            print("Note: Voltage traces not available in CoreNEURON GPU mode")
        
        return spike_data, trace_data
    
    def _setup_spike_recording(self, populations: Dict[str, list]):
        h = self.h
        self._spike_times = h.Vector()
        self._spike_gids = h.Vector()
        self._gid_to_layer = {}
        gid = 0
        
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': self.flatten(populations['L23_E_RS']) + self.flatten(populations['L23_E_FRB']),
            'L5': self.flatten(populations['L5_E_RS']) + self.flatten(populations['L5_E_IB']),
            'L6': self.flatten(populations['L6_E']),
        }
        
        for layer_name, neurons in layer_groups.items():
            for neuron in neurons:
                self.pc.set_gid2node(gid, self.pc.id())
                nc = h.NetCon(neuron.soma(0.5)._ref_v, None, sec=neuron.soma)
                nc.threshold = 0
                self.pc.cell(gid, nc)
                self._gid_to_layer[gid] = layer_name
                gid += 1
        
        self.pc.spike_record(-1, self._spike_times, self._spike_gids)
    
    def _parse_coreneuron_spikes(self, populations: Dict[str, list]) -> Dict[str, np.ndarray]:
        spike_data = {layer: [] for layer in ['thalamus', 'L4', 'L23', 'L5', 'L6']}
        times = np.array(self._spike_times)
        gids = np.array(self._spike_gids, dtype=int)
        
        for t, gid in zip(times, gids):
            if gid in self._gid_to_layer:
                spike_data[self._gid_to_layer[gid]].append(t)
        
        for layer in spike_data:
            spike_data[layer] = np.array(sorted(spike_data[layer]))
        return spike_data
    
    def _extract_all_spikes(self, populations: Dict[str, list]) -> Dict[str, np.ndarray]:
        spike_data = {}
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': self.flatten(populations['L23_E_RS']) + self.flatten(populations['L23_E_FRB']),
            'L5': self.flatten(populations['L5_E_RS']) + self.flatten(populations['L5_E_IB']),
            'L6': self.flatten(populations['L6_E']),
        }
        
        for layer_name, neurons in layer_groups.items():
            all_spikes = []
            for neuron in neurons:
                spikes = self._extract_spike_times(neuron)
                all_spikes.extend(spikes)
            spike_data[layer_name] = np.array(sorted(all_spikes))
        return spike_data
    
    def _extract_traces(self, populations: Dict[str, list]) -> Dict[str, Dict]:
        trace_data = {}
        layer_groups = {
            'thalamus': populations['thalamus_E'],
            'L4': self.flatten(populations['L4_E']),
            'L23': self.flatten(populations['L23_E_RS']) + self.flatten(populations['L23_E_FRB']),
            'L5': self.flatten(populations['L5_E_RS']) + self.flatten(populations['L5_E_IB']),
            'L6': self.flatten(populations['L6_E']),
        }
        
        for layer_name, neurons in layer_groups.items():
            if neurons:
                first_neuron = neurons[0]
                if hasattr(first_neuron, 'vvec') and first_neuron.vvec is not None:
                    trace_data[layer_name] = {
                        'time': np.array(first_neuron.tvec),
                        'voltage': np.array(first_neuron.vvec),
                        'n_neurons': len(neurons)
                    }
        return trace_data if trace_data else None
    
    def _extract_spike_times(self, neuron, threshold: float = 0, refractory_period: float = 2.0) -> List[float]:
        v, t = np.array(neuron.vvec), np.array(neuron.tvec)
        spike_times = []
        last_spike_time = -np.inf
        for i in range(1, len(v)):
            if v[i-1] < threshold <= v[i]:
                if (t[i] - last_spike_time) >= refractory_period:
                    spike_times.append(t[i])
                    last_spike_time = t[i]
        return spike_times
    
    def _mock_run(self) -> Dict[str, np.ndarray]:
        np.random.seed(42)
        p, cfg = self.params, self.network_config
        base_latencies = {'thalamus': 5, 'L4': 12, 'L23': 20, 'L5': 28, 'L6': 35}
        pop_scales = {
            'thalamus': cfg.N_thalamus_E / 5, 'L4': cfg.N_L4_E / 24,
            'L23': cfg.N_L23_E / 24, 'L5': cfg.N_L5_E / 20, 'L6': cfg.N_L6_E / 16,
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
    if path is None:
        default_paths = [
            Path(__file__).parent.parent / 'config' / 'network.yaml',
            Path(__file__).parent.parent / 'config' / 'network.json',
            Path('config/network.yaml'), Path('config/network.json'),
        ]
        for p in default_paths:
            if p.exists():
                return NetworkConfig.from_file(p)
        return NetworkConfig()
    return NetworkConfig.from_file(path)


def run_simulation(params_dict: dict, network_config: Optional[NetworkConfig] = None,
                   config_path: Optional[str] = None, sim_config: Optional[SimulationConfig] = None,
                   use_gpu: bool = False, conn_prob: float = 0.1, seed: Optional[int] = None,
                   return_traces: bool = False) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Optional[Dict]]]:
    import random
    
    if sim_config is None and use_gpu:
        sim_config = SimulationConfig(use_gpu=True)
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    full_params = dict(params_dict)
    full_params['conn_prob'] = conn_prob
    
    params = SimulationParams.from_dict(full_params)
    sim = ThalamoCorticalSimulator(params=params, network_config=network_config,
                                   config_path=config_path, sim_config=sim_config)
    return sim.run(return_traces=return_traces)
