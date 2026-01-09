#!/usr/bin/env python3
"""
Main script for SNN hyperparameter optimization.
Compares Genetic Algorithm vs Bayesian Optimization.
Generates visualizations before and after optimization.

Usage:
    python run_optimization.py --method ga --generations 30 --save-dir results/
    python run_optimization.py --method bo --iterations 50 --save-dir results/
    python run_optimization.py --method compare --budget 50 --save-dir results/
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.parameters import get_parameter_space, get_reduced_parameter_space
from core.objective import ObjectiveFunction
from core.simulator import run_simulation, NetworkConfig, load_config
from core.visualization import SimulationVisualizer, plot_optimization_convergence
from methods.genetic import run_ga, run_nsga3, GAConfig, GeneticAlgorithm
from methods.bayesian import run_bayesian_optimization, BOConfig, BayesianOptimizer


def run_with_visualization(args, optimization_fn, method_name: str):
    """
    Run optimization with before/after visualization.
    
    Args:
        args: CLI arguments
        optimization_fn: Function that runs the optimization and returns result
        method_name: Name of the method for labeling
    """
    # Setup
    param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    network_config = load_config(args.config) if args.config else NetworkConfig()
    
    # Print network config info
    print("\n" + "=" * 60)
    print("NETWORK CONFIGURATION")
    print("=" * 60)
    if args.config:
        print(f"Config file: {args.config}")
    else:
        print("Config file: (using defaults)")
    print(network_config.summary())
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"{method_name}_{timestamp}" if args.save_dir else None
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        visualizer = SimulationVisualizer(save_dir, show=args.show_plots)
        print(f"\nResults will be saved to: {save_dir}")
    else:
        visualizer = None
    
    # Get default parameters
    default_params = param_space.get_default()
    
    print("\n" + "=" * 60)
    print("PHASE 1: Baseline Simulation (Default Parameters)")
    print("=" * 60)
    print(f"\nDefault parameters:")
    for k, v in default_params.items():
        print(f"  {k}: {v}")
    
    # Run simulation with default parameters (with traces)
    print("\nRunning baseline simulation...")
    result = run_simulation(default_params, network_config=network_config, return_traces=True)
    if isinstance(result, tuple):
        spike_data_before, trace_data_before = result
    else:
        spike_data_before = result
        trace_data_before = None
    
    # Compute baseline fitness
    objective = ObjectiveFunction()
    fitness_before = objective(spike_data_before)
    print(f"Baseline fitness: {fitness_before:.4f}")
    
    # Generate before plots
    if visualizer:
        visualizer.plot_all(spike_data_before, prefix="before_", 
                           params_info=default_params, trace_data=trace_data_before)
    
    print("\n" + "=" * 60)
    print(f"PHASE 2: Running {method_name} Optimization")
    print("=" * 60)
    
    # Run optimization (pass network_config)
    result = optimization_fn(args, param_space, network_config=network_config)
    
    print("\n" + "=" * 60)
    print("PHASE 3: Optimized Simulation")
    print("=" * 60)
    print(f"\nOptimized parameters:")
    for k, v in result.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Use stored spike data from optimization (guaranteed to match fitness)
    if hasattr(result, 'best_spike_data') and result.best_spike_data is not None:
        print("\nUsing stored spike data from optimization...")
        spike_data_after = result.best_spike_data
        fitness_after = result.best_fitness
        
        # Run simulation with traces for visualization (with same seed)
        print("Running simulation for voltage traces...")
        result_trace = run_simulation(result.best_params, network_config=network_config, 
                                      return_traces=True, seed=42)
        if isinstance(result_trace, tuple):
            _, trace_data_after = result_trace
        else:
            trace_data_after = None
    else:
        # Fallback: run new simulation (may differ from optimization fitness)
        print("\nRunning optimized simulation...")
        result_after = run_simulation(result.best_params, network_config=network_config, 
                                      return_traces=True, seed=42)
        if isinstance(result_after, tuple):
            spike_data_after, trace_data_after = result_after
        else:
            spike_data_after = result_after
            trace_data_after = None
        fitness_after = objective(spike_data_after)
    
    print(f"Optimized fitness: {fitness_after:.4f}")
    
    # Generate after plots
    if visualizer:
        visualizer.plot_all(spike_data_after, prefix="after_", 
                           params_info=result.best_params, trace_data=trace_data_after)
        
        # Generate comparison plots
        visualizer.plot_comparison(
            spike_data_before, spike_data_after,
            history=result.history,
            params_before=default_params,
            params_after=result.best_params,
            trace_before=trace_data_before,
            trace_after=trace_data_after
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    improvement = (fitness_before - fitness_after) / fitness_before * 100 if fitness_before > 0 else 0
    print(f"\n  Fitness before: {fitness_before:.4f}")
    print(f"  Fitness after:  {fitness_after:.4f}")
    print(f"  Improvement:    {improvement:+.1f}%")
    print(f"  Evaluations:    {result.n_evaluations}")
    print(f"  Runtime:        {result.runtime_seconds:.1f}s")
    
    if save_dir:
        # Save full results
        results_data = {
            'method': method_name,
            'fitness_before': float(fitness_before),
            'fitness_after': float(fitness_after),
            'improvement_percent': float(improvement),
            'params_before': default_params,
            'params_after': result.best_params,
            'history': result.history,
            'n_evaluations': result.n_evaluations,
            'runtime_seconds': result.runtime_seconds,
        }
        
        results_path = save_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"\n  Full results saved to: {results_path}")
    
    return result


def run_genetic_algorithm(args, param_space=None, network_config=None):
    """Run GA optimization."""
    if param_space is None:
        param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    
    return run_ga(
        param_space=param_space,
        n_generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        network_config=network_config,
    )


def run_bayesian(args, param_space=None, network_config=None):
    """Run Bayesian optimization."""
    if param_space is None:
        param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    
    return run_bayesian_optimization(
        param_space=param_space,
        n_iterations=args.iterations,
        n_initial=args.initial,
        seed=args.seed,
        use_optuna=args.use_optuna,
        network_config=network_config,
    )


def run_nsga3_mo(args, param_space=None, network_config=None):
    """Run multi-objective NSGA-III."""
    if param_space is None:
        param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    
    return run_nsga3(
        param_space=param_space,
        n_generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        network_config=network_config,
    )


def run_comparison(args):
    """Compare GA vs BO with same evaluation budget."""
    print("=" * 60)
    print("COMPARISON: GA vs Bayesian Optimization")
    print("=" * 60)
    
    budget = args.budget
    param_space = get_reduced_parameter_space() if args.reduced else get_parameter_space()
    network_config = load_config(args.config) if args.config else NetworkConfig()
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"comparison_{timestamp}" if args.save_dir else None
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be saved to: {save_dir}")
    
    # Get baseline
    default_params = param_space.get_default()
    print("\nRunning baseline simulation...")
    spike_data_baseline = run_simulation(default_params, network_config=network_config)
    objective = ObjectiveFunction()
    fitness_baseline = objective(spike_data_baseline)
    print(f"Baseline fitness: {fitness_baseline:.4f}")
    
    if save_dir:
        vis_baseline = SimulationVisualizer(save_dir / "baseline", show=args.show_plots)
        vis_baseline.plot_all(spike_data_baseline, prefix="baseline_")
    
    # Configure to use approximately same budget
    ga_pop = 20
    ga_gens = budget // ga_pop
    
    bo_initial = 10
    bo_iters = budget
    
    results = {'baseline': {'fitness': fitness_baseline, 'params': default_params}}
    
    # Run GA
    print(f"\n[1/2] Running GA (pop={ga_pop}, gen={ga_gens})...")
    args_ga = argparse.Namespace(
        generations=ga_gens, population=ga_pop, seed=args.seed, reduced=args.reduced
    )
    ga_result = run_genetic_algorithm(args_ga, param_space, network_config=network_config)
    
    spike_data_ga = run_simulation(ga_result.best_params, network_config=network_config)
    fitness_ga = objective(spike_data_ga)
    
    results['ga'] = {
        'result': ga_result,
        'fitness': fitness_ga,
        'spike_data': spike_data_ga,
    }
    
    if save_dir:
        vis_ga = SimulationVisualizer(save_dir / "ga", show=args.show_plots)
        vis_ga.plot_all(spike_data_ga, prefix="ga_")
        vis_ga.plot_comparison(spike_data_baseline, spike_data_ga, 
                               history=ga_result.history,
                               params_before=default_params,
                               params_after=ga_result.best_params)
    
    # Run BO
    print(f"\n[2/2] Running BO (iters={bo_iters}, initial={bo_initial})...")
    args_bo = argparse.Namespace(
        iterations=bo_iters, initial=bo_initial, seed=args.seed, 
        reduced=args.reduced, use_optuna=False
    )
    bo_result = run_bayesian(args_bo, param_space, network_config=network_config)
    
    spike_data_bo = run_simulation(bo_result.best_params, network_config=network_config)
    fitness_bo = objective(spike_data_bo)
    
    results['bo'] = {
        'result': bo_result,
        'fitness': fitness_bo,
        'spike_data': spike_data_bo,
    }
    
    if save_dir:
        vis_bo = SimulationVisualizer(save_dir / "bo", show=args.show_plots)
        vis_bo.plot_all(spike_data_bo, prefix="bo_")
        vis_bo.plot_comparison(spike_data_baseline, spike_data_bo,
                               history=bo_result.history,
                               params_before=default_params,
                               params_after=bo_result.best_params)
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<20} {'Fitness':<15} {'Improvement':<15} {'Evaluations':<15} {'Time (s)':<10}")
    print("-" * 75)
    print(f"{'Baseline':<20} {fitness_baseline:<15.4f} {'--':<15} {'1':<15} {'--':<10}")
    
    ga_imp = (fitness_baseline - fitness_ga) / fitness_baseline * 100 if fitness_baseline > 0 else 0
    bo_imp = (fitness_baseline - fitness_bo) / fitness_baseline * 100 if fitness_baseline > 0 else 0
    
    print(f"{'Genetic Algorithm':<20} {fitness_ga:<15.4f} {ga_imp:<+14.1f}% {ga_result.n_evaluations:<15} {ga_result.runtime_seconds:<10.1f}")
    print(f"{'Bayesian Opt':<20} {fitness_bo:<15.4f} {bo_imp:<+14.1f}% {bo_result.n_evaluations:<15} {bo_result.runtime_seconds:<10.1f}")
    
    # Winner
    if fitness_ga < fitness_bo:
        winner = "Genetic Algorithm"
    elif fitness_bo < fitness_ga:
        winner = "Bayesian Optimization"
    else:
        winner = "Tie"
    
    print(f"\nBest method: {winner}")
    
    # Generate combined comparison plot
    if save_dir:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Raster plots
        for ax, (data, label) in zip(axes, [
            (spike_data_baseline, "Baseline"),
            (spike_data_ga, "GA Optimized"),
            (spike_data_bo, "BO Optimized")
        ]):
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
        
        # Convergence comparison
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # GA convergence
        ga_x = [h['generation'] * ga_pop for h in ga_result.history]
        ga_y = [h['best_fitness'] for h in ga_result.history]
        ax.plot(ga_x, ga_y, 'b-', linewidth=2, label='GA')
        
        # BO convergence
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
        
        # Save summary JSON
        summary = {
            'baseline_fitness': float(fitness_baseline),
            'ga_fitness': float(fitness_ga),
            'ga_improvement': float(ga_imp),
            'ga_params': ga_result.best_params,
            'bo_fitness': float(fitness_bo),
            'bo_improvement': float(bo_imp),
            'bo_params': bo_result.best_params,
            'winner': winner,
        }
        with open(save_dir / "comparison_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Saved: {save_dir / 'comparison_summary.json'}")
    
    return results


def run_grid_search_baseline(args):
    """Run simple grid search as baseline."""
    print("=" * 60)
    print("GRID SEARCH BASELINE")
    print("=" * 60)
    
    param_space = get_reduced_parameter_space()
    objective = ObjectiveFunction()
    
    # Create grid (3 levels per parameter)
    n_levels = 3
    grids = {}
    for spec in param_space.specs:
        if spec.log_scale:
            grids[spec.name] = np.logspace(
                np.log10(spec.min_val), np.log10(spec.max_val), n_levels)
        else:
            grids[spec.name] = np.linspace(spec.min_val, spec.max_val, n_levels)
    
    budget = args.budget
    
    print(f"Parameter space: {param_space.n_params} dimensions")
    print(f"Full grid size: {n_levels ** param_space.n_params}")
    print(f"Sampling {budget} random grid points...")
    
    best_fitness = float('inf')
    best_params = None
    start_time = time.time()
    
    for i in range(budget):
        params = {}
        for name, grid in grids.items():
            params[name] = np.random.choice(grid)
        
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
    
    print(f"\nGrid Search complete!")
    print(f"  Best fitness: {best_fitness:.4f}")
    print(f"  Runtime: {runtime:.1f}s")
    
    return {
        'best_params': best_params,
        'best_fitness': best_fitness,
        'runtime': runtime,
        'n_evaluations': budget,
    }


def main():
    parser = argparse.ArgumentParser(
        description='SNN Hyperparameter Optimization with Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimization.py --method ga --generations 30 --save-dir results/
  python run_optimization.py --method bo --iterations 50 --save-dir results/
  python run_optimization.py --method compare --budget 100 --save-dir results/
  python run_optimization.py --method nsga3 --generations 20 --save-dir results/
        """
    )
    
    parser.add_argument('--method', type=str, default='compare',
                       choices=['ga', 'bo', 'nsga3', 'compare', 'grid'],
                       help='Optimization method')
    
    # GA parameters
    parser.add_argument('--generations', type=int, default=30,
                       help='Number of GA generations')
    parser.add_argument('--population', type=int, default=30,
                       help='GA population size')
    
    # BO parameters
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of BO iterations')
    parser.add_argument('--initial', type=int, default=10,
                       help='Number of initial random samples for BO')
    parser.add_argument('--use-optuna', action='store_true',
                       help='Use Optuna library for BO')
    
    # Common parameters
    parser.add_argument('--budget', type=int, default=100,
                       help='Total evaluation budget for comparison')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--reduced', action='store_true',
                       help='Use reduced parameter space (5 params)')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to network config file (YAML/JSON)')
    
    # Output parameters
    parser.add_argument('--save-dir', '-s', type=str, default='results',
                       help='Directory to save results and plots')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (deprecated, use --save-dir)')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Run selected method
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