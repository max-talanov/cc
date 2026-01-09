"""
Core modules for SNN optimization.
"""

from .simulator import (
    SimulationParams,
    NetworkConfig,
    ThalamoCorticalSimulator,
    run_simulation,
    load_config,
)

from .objective import (
    TargetBehavior,
    ObjectiveFunction,
    MultiObjective,
    evaluate_simulation,
)

from .parameters import (
    ParameterSpec,
    ParameterSpace,
    get_parameter_space,
    get_reduced_parameter_space,
)

__all__ = [
    'SimulationParams',
    'NetworkConfig',
    'ThalamoCorticalSimulator',
    'run_simulation',
    'load_config',
    'TargetBehavior',
    'ObjectiveFunction',
    'MultiObjective',
    'evaluate_simulation',
    'ParameterSpec',
    'ParameterSpace',
    'get_parameter_space',
    'get_reduced_parameter_space',
]