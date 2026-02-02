#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.parameters import get_parameter_space, get_reduced_parameter_space
from core.objective import ObjectiveFunction, create_objective, SupervisedObjective, HybridObjective
from core.simulator import run_simulation, NetworkConfig, load_config, SimulationConfig
from core.visualization import SimulationVisualizer, plot_optimization_convergence
from core.gpu_utils import check_gpu_available, is_gpu_ready
from methods.genetic import run_ga, run_nsga3, GAConfig, GeneticAlgorithm
from methods.bayesian import run_bayesian_optimization, BOConfig, BayesianOptimizer


def _create_objective_from_args(args, sim_duration_ms: float = 100.0):
    objective_type = getattr(args, 'objective', 'rule-based')
    
    if objective_type == 'rule-based':
        return ObjectiveFunction()
    
    data_file = getattr(args, 'data_file', None)
    if data_file is None:
        print(f"Warning: {objective_type} objective requires --data-file. Falling back to rule-based.")
        return ObjectiveFunction()
    
    try:
        from core.data_loader import ExperimentalData
        exp_data = ExperimentalData.from_matlab(data_file, verbose=True)
        print(f"Successfully loaded experimental data: {exp_data.n_trials} trials, "
              f"{exp_data.n_channels} channels, {exp_data.duration_ms:.1f} ms duration")
        print(f"  Simulation duration: {sim_duration_ms:.1f} ms (scaling factor: {exp_data.duration_ms/sim_duration_ms:.1f}x)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Falling back to rule-based objective.")
        return ObjectiveFunction()
    
    trial_idx = getattr(args, 'trial_idx', None)
    
    if objective_type == 'supervised':
        return SupervisedObjective(experimental_data=exp_data, trial_idx=trial_idx,
                                   duration_ms=sim_duration_ms)
    elif objective_type == 'hybrid':
        return HybridObjective(experimental_data=exp_data,
                              supervised_weight=getattr(args, 'supervised_weight', 0.5), 
                              trial_idx=trial_idx, duration_ms=sim_duration_ms)
    return ObjectiveFunction()


def _get_sim_config(args):
    return SimulationConfig(
        use_gpu=getattr(args, 'gpu', False),
        gpu_device=getattr(args, 'gpu_device', 0),
        num_threads=getattr(args, 'threads', 4),
        use_mpi=getattr(args, 'mpi', False),
        cache_efficient=True, verbose=True
    )


def run_with_visualization(args, optimization_fn, method_name: str):
    param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    network_config = load_config(args.config) if args.config else NetworkConfig()
    sim_config = _get_sim_config(args)
    
    print("\n" + "=" * 60 + "\nEXECUTION MODE\n" + "=" * 60)
    if sim_config.use_gpu:
        print("GPU MODE ENABLED")
        available, msg = check_gpu_available()
        print(f"GPU Status: {msg}")
        if not available:
            print("Warning: GPU not available, falling back to CPU")
            sim_config.use_gpu = False
    elif sim_config.use_mpi:
        print("MPI MODE ENABLED\nLaunch with: mpirun -n N python run_opt.py ...")
    else:
        print(f"CPU MODE: {sim_config.num_threads} threads")
    
    # Pass simulation duration to objective for proper scaling
    objective = _create_objective_from_args(args, sim_duration_ms=network_config.tstop)
    
    print("\n" + "=" * 60 + "\nNETWORK CONFIGURATION\n" + "=" * 60)
    print(f"Config file: {args.config if args.config else '(using defaults)'}")
    print(network_config.summary())
    
    print("\n" + "=" * 60 + "\nOBJECTIVE FUNCTION\n" + "=" * 60)
    print(f"Type: {args.objective}")
    if args.objective != 'rule-based':
        print(f"Data file: {args.data_file}")
        if args.objective == 'hybrid':
            print(f"Supervised weight: {args.supervised_weight}")
    
    args._sim_config = sim_config
    args._objective = objective
    args._conn_prob = getattr(args, 'conn_prob', 0.1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"{method_name}_{timestamp}" if args.save_dir else None
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        visualizer = SimulationVisualizer(save_dir, show=args.show_plots)
        print(f"\nResults will be saved to: {save_dir}")
    else:
        visualizer = None
    
    default_params = param_space.get_default()
    
    print("\n" + "=" * 60 + "\nPHASE 1: Baseline Simulation (Default Parameters)\n" + "=" * 60)
    print(f"\nDefault parameters:")
    for k, v in default_params.items():
        print(f"  {k}: {v}")
    
    print("\nRunning baseline simulation...")
    spike_data_before, trace_data_before = run_simulation(
        default_params, network_config=network_config, sim_config=sim_config,
        conn_prob=args._conn_prob, return_traces=True
    )
    
    fitness_before = objective(spike_data_before)
    print(f"Baseline fitness: {fitness_before:.4f}")
    
    if visualizer:
        visualizer.plot_all(spike_data_before, prefix="before_", params_info=default_params, trace_data=trace_data_before)
    
    print("\n" + "=" * 60 + f"\nPHASE 2: Running {method_name} Optimization\n" + "=" * 60)
    result = optimization_fn(args, param_space, network_config=network_config)
    
    print("\n" + "=" * 60 + "\nPHASE 3: Optimized Simulation\n" + "=" * 60)
    print(f"\nOptimized parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    
    if hasattr(result, 'best_spike_data') and result.best_spike_data is not None:
        print("\nUsing stored spike data from optimization...")
        spike_data_after = result.best_spike_data
        fitness_after = result.best_fitness
        print("Running simulation for voltage traces...")
        _, trace_data_after = run_simulation(result.best_params, network_config=network_config,
                                             sim_config=sim_config, conn_prob=args._conn_prob,
                                             return_traces=True, seed=42)
    else:
        print("\nRunning optimized simulation...")
        spike_data_after, trace_data_after = run_simulation(
            result.best_params, network_config=network_config, sim_config=sim_config,
            conn_prob=args._conn_prob, return_traces=True, seed=42
        )
        fitness_after = objective(spike_data_after)
    
    print(f"Optimized fitness: {fitness_after:.4f}")
    
    if visualizer:
        visualizer.plot_all(spike_data_after, prefix="after_", params_info=result.best_params, trace_data=trace_data_after)
        visualizer.plot_comparison(spike_data_before, spike_data_after, history=result.history,
                                   params_before=default_params, params_after=result.best_params,
                                   trace_before=trace_data_before, trace_after=trace_data_after)
    
    print("\n" + "=" * 60 + "\nOPTIMIZATION SUMMARY\n" + "=" * 60)
    improvement = (fitness_before - fitness_after) / fitness_before * 100 if fitness_before > 0 else 0
    print(f"\n  Fitness before: {fitness_before:.4f}\n  Fitness after:  {fitness_after:.4f}\n"
          f"  Improvement:    {improvement:+.1f}%\n  Evaluations:    {result.n_evaluations}\n"
          f"  Runtime:        {result.runtime_seconds:.1f}s")
    
    if save_dir:
        results_data = {
            'method': method_name, 'fitness_before': float(fitness_before),
            'fitness_after': float(fitness_after), 'improvement_percent': float(improvement),
            'params_before': default_params, 'params_after': result.best_params,
            'history': result.history, 'n_evaluations': result.n_evaluations,
            'runtime_seconds': result.runtime_seconds,
        }
        results_path = save_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\n  Full results saved to: {results_path}")
    
    return result


def run_genetic_algorithm(args, param_space=None, network_config=None):
    if param_space is None:
        param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    
    sim_config = getattr(args, '_sim_config', None)
    objective_fn = getattr(args, '_objective', None)
    conn_prob = getattr(args, '_conn_prob', getattr(args, 'conn_prob', 0.1))
    
    return run_ga(param_space=param_space, n_generations=args.generations, population_size=args.population,
                  seed=args.seed, network_config=network_config, sim_config=sim_config,
                  objective_fn=objective_fn, conn_prob=conn_prob)


def run_bayesian(args, param_space=None, network_config=None):
    if param_space is None:
        param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    
    sim_config = getattr(args, '_sim_config', None)
    objective_fn = getattr(args, '_objective', None)
    conn_prob = getattr(args, '_conn_prob', getattr(args, 'conn_prob', 0.1))
    
    return run_bayesian_optimization(param_space=param_space, n_iterations=args.iterations,
                                     n_initial=args.initial, seed=args.seed, use_optuna=args.use_optuna,
                                     network_config=network_config, sim_config=sim_config,
                                     objective_fn=objective_fn, conn_prob=conn_prob)


def run_nsga3_mo(args, param_space=None, network_config=None):
    if param_space is None:
        param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    sim_config = getattr(args, '_sim_config', None)
    return run_nsga3(param_space=param_space, n_generations=args.generations,
                     population_size=args.population, seed=args.seed,
                     network_config=network_config, sim_config=sim_config)


def run_comparison(args):
    print("=" * 60 + "\nCOMPARISON: GA vs Bayesian Optimization\n" + "=" * 60)
    
    budget = args.budget
    param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    network_config = load_config(args.config) if args.config else NetworkConfig()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"comparison_{timestamp}" if args.save_dir else None
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be saved to: {save_dir}")
    
    default_params = param_space.get_default()
    print("\nRunning baseline simulation...")
    spike_data_baseline = run_simulation(default_params, network_config=network_config)
    objective = ObjectiveFunction()
    fitness_baseline = objective(spike_data_baseline)
    print(f"Baseline fitness: {fitness_baseline:.4f}")
    
    if save_dir:
        vis_baseline = SimulationVisualizer(save_dir / "baseline", show=args.show_plots)
        vis_baseline.plot_all(spike_data_baseline, prefix="baseline_")
    
    ga_pop, ga_gens = 20, budget // 20
    bo_initial, bo_iters = 10, budget
    
    results = {'baseline': {'fitness': fitness_baseline, 'params': default_params}}
    
    print(f"\n[1/2] Running GA (pop={ga_pop}, gen={ga_gens})...")
    args_ga = argparse.Namespace(generations=ga_gens, population=ga_pop, seed=args.seed, reduced=args.reduced)
    ga_result = run_genetic_algorithm(args_ga, param_space, network_config=network_config)
    
    spike_data_ga = run_simulation(ga_result.best_params, network_config=network_config)
    fitness_ga = objective(spike_data_ga)
    results['ga'] = {'result': ga_result, 'fitness': fitness_ga, 'spike_data': spike_data_ga}
    
    if save_dir:
        vis_ga = SimulationVisualizer(save_dir / "ga", show=args.show_plots)
        vis_ga.plot_all(spike_data_ga, prefix="ga_")
        vis_ga.plot_comparison(spike_data_baseline, spike_data_ga, history=ga_result.history,
                               params_before=default_params, params_after=ga_result.best_params)
    
    print(f"\n[2/2] Running BO (iters={bo_iters}, initial={bo_initial})...")
    args_bo = argparse.Namespace(iterations=bo_iters, initial=bo_initial, seed=args.seed,
                                  reduced=args.reduced, use_optuna=False)
    bo_result = run_bayesian(args_bo, param_space, network_config=network_config)
    
    spike_data_bo = run_simulation(bo_result.best_params, network_config=network_config)
    fitness_bo = objective(spike_data_bo)
    results['bo'] = {'result': bo_result, 'fitness': fitness_bo, 'spike_data': spike_data_bo}
    
    if save_dir:
        vis_bo = SimulationVisualizer(save_dir / "bo", show=args.show_plots)
        vis_bo.plot_all(spike_data_bo, prefix="bo_")
        vis_bo.plot_comparison(spike_data_baseline, spike_data_bo, history=bo_result.history,
                               params_before=default_params, params_after=bo_result.best_params)
    
    print("\n" + "=" * 60 + "\nCOMPARISON SUMMARY\n" + "=" * 60)
    print(f"\n{'Method':<20} {'Fitness':<15} {'Improvement':<15} {'Evaluations':<15} {'Time (s)':<10}")
    print("-" * 75)
    print(f"{'Baseline':<20} {fitness_baseline:<15.4f} {'--':<15} {'1':<15} {'--':<10}")
    
    ga_imp = (fitness_baseline - fitness_ga) / fitness_baseline * 100 if fitness_baseline > 0 else 0
    bo_imp = (fitness_baseline - fitness_bo) / fitness_baseline * 100 if fitness_baseline > 0 else 0
    
    print(f"{'Genetic Algorithm':<20} {fitness_ga:<15.4f} {ga_imp:<+14.1f}% {ga_result.n_evaluations:<15} {ga_result.runtime_seconds:<10.1f}")
    print(f"{'Bayesian Opt':<20} {fitness_bo:<15.4f} {bo_imp:<+14.1f}% {bo_result.n_evaluations:<15} {bo_result.runtime_seconds:<10.1f}")
    
    if fitness_ga < fitness_bo:
        winner = "Genetic Algorithm"
    elif fitness_bo < fitness_ga:
        winner = "Bayesian Optimization"
    else:
        winner = "Tie"
    print(f"\nBest method: {winner}")
    
    if save_dir:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (data, label) in zip(axes, [(spike_data_baseline, "Baseline"),
                                             (spike_data_ga, "GA Optimized"),
                                             (spike_data_bo, "BO Optimized")]):
            layers = list(data.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
            for idx, layer in enumerate(layers):
                spikes = data[layer]
                spikes = spikes[(spikes >= 0) & (spikes <= 100)]
                if len(spikes) > 0:
                    ax.scatter(spikes, [idx] * len(spikes), s=10, alpha=0.7, color=colors[idx])
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Layer")
            ax.set_title(label)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig.savefig(save_dir / "comparison_all_methods.png", dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {save_dir / 'comparison_all_methods.png'}")
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ga_x = [h['generation'] * ga_pop for h in ga_result.history]
        ga_y = [h['best_fitness'] for h in ga_result.history]
        ax.plot(ga_x, ga_y, 'b-', linewidth=2, label='GA')
        bo_x = [h['iteration'] for h in bo_result.history]
        bo_y = [h['best_fitness'] for h in bo_result.history]
        ax.plot(bo_x, bo_y, 'r-', linewidth=2, label='BO')
        ax.axhline(y=fitness_baseline, color='gray', linestyle='--', label='Baseline')
        ax.set_xlabel("Evaluations (approx)")
        ax.set_ylabel("Best Fitness")
        ax.set_title("Convergence Comparison: GA vs BO")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_dir / "convergence_comparison.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_dir / 'convergence_comparison.png'}")
        if args.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        summary = {
            'baseline_fitness': float(fitness_baseline),
            'ga_fitness': float(fitness_ga), 'ga_improvement': float(ga_imp), 'ga_params': ga_result.best_params,
            'bo_fitness': float(fitness_bo), 'bo_improvement': float(bo_imp), 'bo_params': bo_result.best_params,
            'winner': winner,
        }
        with open(save_dir / "comparison_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Saved: {save_dir / 'comparison_summary.json'}")
    
    return results


def run_grid_search_baseline(args):
    print("=" * 60 + "\nGRID SEARCH BASELINE\n" + "=" * 60)
    
    param_space = get_reduced_parameter_space()
    objective = ObjectiveFunction()
    n_levels = 3
    
    grids = {}
    for spec in param_space.specs:
        if spec.log_scale:
            grids[spec.name] = np.logspace(np.log10(spec.min_val), np.log10(spec.max_val), n_levels)
        else:
            grids[spec.name] = np.linspace(spec.min_val, spec.max_val, n_levels)
    
    budget = args.budget
    print(f"Parameter space: {param_space.n_params} dimensions")
    print(f"Full grid size: {n_levels ** param_space.n_params}")
    print(f"Sampling {budget} random grid points...")
    
    best_fitness, best_params = float('inf'), None
    start_time = time.time()
    
    for i in range(budget):
        params = {name: np.random.choice(grid) for name, grid in grids.items()}
        try:
            spike_data = run_simulation(params)
            fitness = objective(spike_data)
        except:
            fitness = 1e6
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_params = params.copy()
            print(f"  {i+1}/{budget}: NEW BEST {fitness:.4f}")
        elif i % 10 == 0:
            print(f"  {i+1}/{budget}: {fitness:.4f}")
    
    runtime = time.time() - start_time
    print(f"\nGrid Search complete!\n  Best fitness: {best_fitness:.4f}\n  Runtime: {runtime:.1f}s")
    
    return {'best_params': best_params, 'best_fitness': best_fitness, 'runtime': runtime, 'n_evaluations': budget}


def main():
    parser = argparse.ArgumentParser(description='SNN Hyperparameter Optimization')
    
    parser.add_argument('--method', type=str, default='compare',
                       choices=['ga', 'bo', 'nsga3', 'compare', 'grid'])
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--population', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--initial', type=int, default=10)
    parser.add_argument('--use-optuna', action='store_true')
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--save-dir', '-s', type=str, default='results')
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--threads', '-t', type=int, default=4)
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--conn-prob', type=float, default=0.1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--check-gpu', action='store_true')
    parser.add_argument('--objective', type=str, default='rule-based',
                       choices=['rule-based', 'supervised', 'hybrid'])
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--supervised-weight', type=float, default=0.5)
    parser.add_argument('--trial-idx', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.check_gpu:
        available, message = check_gpu_available()
        print(f"GPU Available: {available}\nDetails: {message}")
        return
    
    if args.method == 'ga':
        run_with_visualization(args, run_genetic_algorithm, "ga")
    elif args.method == 'bo':
        run_with_visualization(args, run_bayesian, "bo")
    elif args.method == 'nsga3':
        run_with_visualization(args, run_nsga3_mo, "nsga3")
    elif args.method == 'compare':
        run_comparison(args)
    elif args.method == 'grid':
        run_grid_search_baseline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
