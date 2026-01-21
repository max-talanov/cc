"""Visualization module for thalamo-cortical simulation results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime


def plot_raster(spike_data: Dict[str, np.ndarray], t_start: Optional[float] = None,
                t_end: Optional[float] = None, tick_step: float = 10,
                title: Optional[str] = None, figsize: tuple = (14, 8),
                save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    group_names = list(spike_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))
    
    for idx, (group_name, spikes) in enumerate(spike_data.items()):
        if len(spikes) == 0:
            continue
        filtered = spikes.copy()
        if t_start is not None:
            filtered = filtered[filtered >= t_start]
        if t_end is not None:
            filtered = filtered[filtered <= t_end]
        if len(filtered) == 0:
            continue
        ax.scatter(filtered, [idx] * len(filtered), s=15, alpha=0.7, 
                   color=colors[idx], label=group_name)
    
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title or f"Spike Raster ({t_start or 0:.0f} – {t_end or 'end'} ms)", fontsize=14)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_spike_histogram(spike_data: Dict[str, np.ndarray], layer: str = 'L4',
                         t_start: float = 0, t_end: float = 200, bin_size: float = 5,
                         title: Optional[str] = None, figsize: tuple = (12, 5),
                         save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    if layer not in spike_data:
        ax.text(0.5, 0.5, f"No data for layer {layer}", transform=ax.transAxes, ha='center')
        return fig
    
    spikes = spike_data[layer]
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    counts, edges = np.histogram(spikes, bins=bins)
    
    ax.bar(edges[:-1], counts, width=bin_size, align='edge', 
           color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Spike count", fontsize=12)
    ax.set_title(title or f"Spike Histogram - {layer}", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_spike_heatmap(spike_data: Dict[str, np.ndarray], t_start: float = 0,
                       t_end: float = 100, bin_size: float = 5, title: Optional[str] = None,
                       figsize: tuple = (14, 6), save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    group_names = list(spike_data.keys())
    
    hist_matrix = []
    for layer in group_names:
        counts, _ = np.histogram(spike_data[layer], bins=bins)
        hist_matrix.append(counts)
    hist_matrix = np.array(hist_matrix)
    
    im = ax.imshow(hist_matrix, aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')
    plt.colorbar(im, ax=ax, label="Spike count")
    
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title or "Spike Activity Heatmap", fontsize=14)
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    n_ticks = min(10, len(bin_centers))
    tick_idx = np.linspace(0, len(bin_centers)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{bin_centers[i]:.0f}" for i in tick_idx])
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_latency_comparison(spike_data_before: Dict[str, np.ndarray],
                            spike_data_after: Dict[str, np.ndarray],
                            title: Optional[str] = None, figsize: tuple = (12, 6),
                            save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    layers = list(spike_data_before.keys())
    x = np.arange(len(layers))
    width = 0.35
    
    def get_first_spike(spikes):
        return spikes[0] if len(spikes) > 0 else np.nan
    
    latencies_before = [get_first_spike(spike_data_before[l]) for l in layers]
    latencies_after = [get_first_spike(spike_data_after[l]) for l in layers]
    
    bars1 = ax.bar(x - width/2, latencies_before, width, label='Before', 
                   color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, latencies_after, width, label='After', 
                   color='lightgreen', edgecolor='black')
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("First Spike Latency (ms)", fontsize=12)
    ax.set_title(title or "First Spike Latency Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, latencies_before):
        if not np.isnan(val):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, latencies_after):
        if not np.isnan(val):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_firing_rate_comparison(spike_data_before: Dict[str, np.ndarray],
                                spike_data_after: Dict[str, np.ndarray],
                                duration: float = 200.0, title: Optional[str] = None,
                                figsize: tuple = (12, 6), save_path: Optional[str] = None,
                                show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    layers = list(spike_data_before.keys())
    x = np.arange(len(layers))
    width = 0.35
    
    def get_rate(spikes):
        return len(spikes) / (duration / 1000)
    
    rates_before = [get_rate(spike_data_before[l]) for l in layers]
    rates_after = [get_rate(spike_data_after[l]) for l in layers]
    
    ax.bar(x - width/2, rates_before, width, label='Before', color='lightcoral', edgecolor='black')
    ax.bar(x + width/2, rates_after, width, label='After', color='lightgreen', edgecolor='black')
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
    ax.set_title(title or "Firing Rate Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_optimization_convergence(history: List[Dict], title: Optional[str] = None,
                                  figsize: tuple = (12, 5), save_path: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    x_key = 'generation' if 'generation' in history[0] else 'iteration'
    x_label = 'Generation' if x_key == 'generation' else 'Iteration'
    
    x = [h[x_key] for h in history]
    best = [h['best_fitness'] for h in history]
    
    ax.plot(x, best, 'b-', linewidth=2, label='Best fitness')
    
    if 'mean_fitness' in history[0]:
        mean = [h['mean_fitness'] for h in history]
        ax.plot(x, mean, 'r--', alpha=0.7, label='Mean fitness')
        ax.fill_between(x, best, mean, alpha=0.2, color='blue')
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Fitness (lower is better)", fontsize=12)
    ax.set_title(title or "Optimization Convergence", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_voltage_traces(trace_data: Dict, t_max: float = 100, title: Optional[str] = None,
                        figsize: tuple = (14, 10), save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    if trace_data is None or len(trace_data) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No voltage trace data available\n(enable with return_traces=True)",
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if not show:
            plt.close(fig)
        return fig
    
    layers = list(trace_data.keys())
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, sharex=True)
    if n_layers == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_layers))
    
    for idx, (layer, data) in enumerate(trace_data.items()):
        ax = axes[idx]
        t = data.get('time', np.array([]))
        v = data.get('voltage', np.array([]))
        
        if len(t) == 0 or len(v) == 0:
            ax.text(0.5, 0.5, f"No data for {layer}", transform=ax.transAxes, ha='center', va='center')
            ax.set_ylabel(layer)
            continue
        
        mask = t <= t_max
        t_masked = t[mask]
        v_masked = v[:len(t_masked)]
        
        ax.plot(t_masked, v_masked, color=colors[idx], linewidth=0.8, alpha=0.9)
        ax.set_ylabel(f"{layer}\n(mV)", fontsize=10)
        ax.set_ylim(-80, 50)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        spike_mask = v_masked > 0
        if np.any(spike_mask):
            ax.scatter(t_masked[spike_mask], v_masked[spike_mask], color='red', s=10, alpha=0.7, zorder=5)
    
    axes[-1].set_xlabel("Time (ms)", fontsize=12)
    axes[0].set_title(title or "Voltage Traces by Layer", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_membrane_potential(trace_data: Dict, group_type: str = 'excitatory',
                            t_max: float = 150, title: Optional[str] = None,
                            figsize: tuple = (14, 7), save_path: Optional[str] = None,
                            show: bool = True) -> plt.Figure:
    if trace_data and isinstance(list(trace_data.values())[0], dict):
        first_val = list(trace_data.values())[0]
        if 'voltage' in first_val:
            return plot_voltage_traces(trace_data, t_max=t_max, title=title,
                                       figsize=figsize, save_path=save_path, show=show)
    
    fig, ax = plt.subplots(figsize=figsize)
    t = trace_data.get('time')
    groups = trace_data.get(group_type, {})
    
    if t is None or len(groups) == 0:
        ax.text(0.5, 0.5, f"No {group_type} trace data available",
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        return fig
    
    mask = t <= t_max
    t_masked = t[mask]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    for (group_name, v), color in zip(groups.items(), colors):
        v_masked = v[mask] if len(v) > sum(mask) else v[:sum(mask)]
        ax.plot(t_masked[:len(v_masked)], v_masked, label=group_name, color=color, linewidth=1.2, alpha=0.8)
    
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Membrane Potential (mV)", fontsize=12)
    ax.set_title(title or f"Membrane Potential ({group_type.capitalize()} Neurons)", fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)
    ax.set_ylim(-90, 50)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_membrane_potential_comparison(trace_before: Dict, trace_after: Dict,
                                       group_type: str = 'excitatory', t_max: float = 100,
                                       title: Optional[str] = None, figsize: tuple = (16, 8),
                                       save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    def is_new_format(trace_data):
        if trace_data is None or len(trace_data) == 0:
            return False
        first_val = list(trace_data.values())[0]
        return isinstance(first_val, dict) and 'voltage' in first_val
    
    if is_new_format(trace_before) or is_new_format(trace_after):
        return _plot_voltage_traces_comparison(trace_before, trace_after, group_type=group_type,
                                               t_max=t_max, title=title, figsize=figsize,
                                               save_path=save_path, show=show)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    
    for ax, (trace_data, label) in zip(axes, [(trace_before, "Before Optimization"),
                                               (trace_after, "After Optimization")]):
        t = trace_data.get('time')
        groups = trace_data.get(group_type, {})
        
        if t is None or len(groups) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')
            ax.set_title(label)
            continue
        
        mask = t <= t_max
        t_masked = t[mask]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for (group_name, v), color in zip(groups.items(), colors):
            v_masked = v[mask] if len(v) > sum(mask) else v[:sum(mask)]
            ax.plot(t_masked[:len(v_masked)], v_masked, label=group_name, color=color, linewidth=1.0, alpha=0.8)
        
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_max)
        ax.legend(loc='upper right', fontsize=8)
    
    axes[0].set_ylabel("Membrane Potential (mV)", fontsize=11)
    fig.suptitle(title or f"{group_type.capitalize()} Membrane Potentials: Before vs After", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _plot_voltage_traces_comparison(trace_before: Dict, trace_after: Dict,
                                    group_type: str = 'excitatory', t_max: float = 100,
                                    title: Optional[str] = None, figsize: tuple = (16, 10),
                                    save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    excitatory_layers = ['thalamus', 'L4', 'L23', 'L5', 'L6']
    target_layers = excitatory_layers
    
    if (trace_before is None or len(trace_before) == 0) and \
       (trace_after is None or len(trace_after) == 0):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No {group_type} trace data available\n(enable with return_traces=True)",
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        if not show:
            plt.close(fig)
        return fig
    
    available_layers = set()
    if trace_before:
        available_layers.update(trace_before.keys())
    if trace_after:
        available_layers.update(trace_after.keys())
    
    layers = [l for l in target_layers if l in available_layers]
    
    if len(layers) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No {group_type} trace data available",
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if not show:
            plt.close(fig)
        return fig
    
    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=figsize, sharex=True, sharey='row')
    if n_layers == 1:
        axes = axes.reshape(1, 2)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_layers))
    
    for idx, layer in enumerate(layers):
        for col, (trace_data, label) in enumerate([(trace_before, "Before"), (trace_after, "After")]):
            ax = axes[idx, col]
            
            if trace_data is None or layer not in trace_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center', fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{layer}\n(mV)", fontsize=10)
                if idx == 0:
                    ax.set_title(f"{label} Optimization", fontsize=12)
                continue
            
            data = trace_data[layer]
            t = data.get('time', np.array([]))
            v = data.get('voltage', np.array([]))
            
            if len(t) == 0 or len(v) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center', fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{layer}\n(mV)", fontsize=10)
                if idx == 0:
                    ax.set_title(f"{label} Optimization", fontsize=12)
                continue
            
            mask = t <= t_max
            t_masked = t[mask]
            v_masked = v[:len(t_masked)] if len(v) >= len(t_masked) else v
            
            ax.plot(t_masked[:len(v_masked)], v_masked, color=colors[idx], linewidth=0.8, alpha=0.9)
            ax.set_ylim(-80, 50)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            spike_mask = v_masked > 0
            if np.any(spike_mask):
                t_plot = t_masked[:len(v_masked)]
                ax.scatter(t_plot[spike_mask], v_masked[spike_mask], color='red', s=10, alpha=0.7, zorder=5)
            
            if col == 0:
                ax.set_ylabel(f"{layer}\n(mV)", fontsize=10)
            if idx == 0:
                ax.set_title(f"{label} Optimization", fontsize=12)
    
    for col in range(2):
        axes[-1, col].set_xlabel("Time (ms)", fontsize=11)
    
    fig.suptitle(title or f"{group_type.capitalize()} Neurons: Before vs After Optimization", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_single_layer_neurons(trace_data: Dict, layer: str = 'L4 Spiny Stellate',
                              t_max: float = 100, title: Optional[str] = None,
                              figsize: tuple = (12, 6), save_path: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    t = trace_data.get('time')
    
    v = None
    for group_type in ['excitatory', 'inhibitory']:
        groups = trace_data.get(group_type, {})
        if layer in groups:
            v = groups[layer]
            break
    
    if t is None or v is None:
        ax.text(0.5, 0.5, f"No data for {layer}", transform=ax.transAxes, ha='center', va='center', fontsize=14)
        return fig
    
    mask = t <= t_max
    t_masked = t[mask]
    v_masked = v[mask] if len(v) > sum(mask) else v[:sum(mask)]
    
    ax.plot(t_masked[:len(v_masked)], v_masked, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Membrane Potential (mV)", fontsize=12)
    ax.set_title(title or f"Membrane Potential - {layer}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


class SimulationVisualizer:
    """Convenience class for generating all visualizations."""
    
    def __init__(self, save_dir: Union[str, Path], show: bool = False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_all(self, spike_data: Dict[str, np.ndarray], prefix: str = "",
                 params_info: Optional[Dict] = None, trace_data: Optional[Dict] = None):
        print(f"\nGenerating plots with prefix '{prefix}'...")
        
        plot_raster(spike_data, t_start=0, t_end=100,
                   title=f"{prefix.replace('_', ' ').title()}Spike Raster (0-100ms)",
                   save_path=str(self.save_dir / f"{prefix}raster.png"), show=self.show)
        
        plot_spike_heatmap(spike_data, t_start=0, t_end=100, bin_size=5,
                          title=f"{prefix.replace('_', ' ').title()}Activity Heatmap",
                          save_path=str(self.save_dir / f"{prefix}heatmap.png"), show=self.show)
        
        for layer in spike_data.keys():
            plot_spike_histogram(spike_data, layer=layer, t_start=0, t_end=100, bin_size=5,
                               title=f"{prefix.replace('_', ' ').title()}{layer} Spike Histogram",
                               save_path=str(self.save_dir / f"{prefix}histogram_{layer}.png"), show=self.show)
        
        if trace_data is not None:
            plot_membrane_potential(trace_data, group_type='excitatory', t_max=150,
                                   title=f"{prefix.replace('_', ' ').title()}Excitatory Membrane Potentials",
                                   save_path=str(self.save_dir / f"{prefix}membrane_excitatory.png"), show=self.show)
            plot_membrane_potential(trace_data, group_type='inhibitory', t_max=150,
                                   title=f"{prefix.replace('_', ' ').title()}Inhibitory Membrane Potentials",
                                   save_path=str(self.save_dir / f"{prefix}membrane_inhibitory.png"), show=self.show)
    
    def plot_comparison(self, spike_data_before: Dict[str, np.ndarray],
                        spike_data_after: Dict[str, np.ndarray],
                        history: Optional[List[Dict]] = None,
                        params_before: Optional[Dict] = None,
                        params_after: Optional[Dict] = None,
                        trace_before: Optional[Dict] = None,
                        trace_after: Optional[Dict] = None):
        print("\nGenerating comparison plots...")
        
        plot_latency_comparison(spike_data_before, spike_data_after,
                               title="First Spike Latency: Before vs After Optimization",
                               save_path=str(self.save_dir / "comparison_latency.png"), show=self.show)
        
        plot_firing_rate_comparison(spike_data_before, spike_data_after,
                                   title="Firing Rate: Before vs After Optimization",
                                   save_path=str(self.save_dir / "comparison_firing_rate.png"), show=self.show)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        for ax, (spike_data, label) in zip(axes, [(spike_data_before, "Before"), (spike_data_after, "After")]):
            layers = list(spike_data.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
            for idx, layer in enumerate(layers):
                spikes = spike_data[layer]
                spikes = spikes[(spikes >= 0) & (spikes <= 100)]
                if len(spikes) > 0:
                    ax.scatter(spikes, [idx] * len(spikes), s=15, alpha=0.7, color=colors[idx])
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.set_ylabel("Layer", fontsize=12)
            ax.set_title(f"{label} Optimization", fontsize=14)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig.savefig(str(self.save_dir / "comparison_raster_side_by_side.png"), dpi=150, bbox_inches='tight')
        print(f"  Saved: {self.save_dir / 'comparison_raster_side_by_side.png'}")
        if self.show:
            plt.show()
        else:
            plt.close(fig)
        
        if history:
            plot_optimization_convergence(history, title="Optimization Convergence",
                                         save_path=str(self.save_dir / "convergence.png"), show=self.show)
        
        if trace_before is not None and trace_after is not None:
            plot_membrane_potential_comparison(trace_before, trace_after, group_type='excitatory', t_max=100,
                                              title="Excitatory Neurons: Before vs After Optimization",
                                              save_path=str(self.save_dir / "comparison_membrane_excitatory.png"),
                                              show=self.show)
            plot_membrane_potential_comparison(trace_before, trace_after, group_type='inhibitory', t_max=100,
                                              title="Inhibitory Neurons: Before vs After Optimization",
                                              save_path=str(self.save_dir / "comparison_membrane_inhibitory.png"),
                                              show=self.show)
        
        if params_before or params_after:
            self._save_params_summary(params_before, params_after)
    
    def _save_params_summary(self, params_before: Optional[Dict], params_after: Optional[Dict]):
        summary_path = self.save_dir / "parameters_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\nOPTIMIZATION PARAMETERS SUMMARY\n" + "=" * 60 + "\n\n")
            if params_before:
                f.write("BEFORE (Default Parameters):\n" + "-" * 40 + "\n")
                for k, v in params_before.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
            if params_after:
                f.write("AFTER (Optimized Parameters):\n" + "-" * 40 + "\n")
                for k, v in params_after.items():
                    f.write(f"  {k}: {v:.6f}\n" if isinstance(v, float) else f"  {k}: {v}\n")
                f.write("\n")
            if params_before and params_after:
                f.write("CHANGES:\n" + "-" * 40 + "\n")
                for k in params_after:
                    if k in params_before:
                        before_val, after_val = params_before[k], params_after[k]
                        if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                            if before_val != 0:
                                pct_change = (after_val - before_val) / abs(before_val) * 100
                                f.write(f"  {k}: {before_val} → {after_val:.6f} ({pct_change:+.1f}%)\n")
                            else:
                                f.write(f"  {k}: {before_val} → {after_val:.6f}\n")
        print(f"  Saved: {summary_path}")
