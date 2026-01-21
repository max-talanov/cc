"""
Core modules for SNN optimization.
"""

from .simulator import (
    SimulationParams,
    SimulationConfig,
    NetworkConfig,
    ThalamoCorticalSimulator,
    run_simulation,
    load_config,
)

from .objective import (
    TargetBehavior,
    ObjectiveFunction,
    MultiObjective,
    SupervisedObjective,
    HybridObjective,
    evaluate_simulation,
    create_objective,
)

from .parameters import (
    ParameterSpec,
    ParameterSpace,
    get_parameter_space,
    get_reduced_parameter_space,
)

from .data_loader import (
    ExperimentalData,
    create_channel_mapping,
)

from .gpu_utils import (
    check_coreneuron_available,
    check_gpu_available,
    get_gpu_info,
    is_gpu_ready,
    parse_spike_file,
    aggregate_spikes_by_layer,
)

__all__ = [
    # Simulator
    'SimulationParams',
    'SimulationConfig',
    'NetworkConfig',
    'ThalamoCorticalSimulator',
    'run_simulation',
    'load_config',
    # Objectives
    'TargetBehavior',
    'ObjectiveFunction',
    'MultiObjective',
    'SupervisedObjective',
    'HybridObjective',
    'evaluate_simulation',
    'create_objective',
    # Parameters
    'ParameterSpec',
    'ParameterSpace',
    'get_parameter_space',
    'get_reduced_parameter_space',
    # Data loading
    'ExperimentalData',
    'create_channel_mapping',
    # GPU utilities
    'check_coreneuron_available',
    'check_gpu_available',
    'get_gpu_info',
    'is_gpu_ready',
    'parse_spike_file',
    'aggregate_spikes_by_layer',
]
