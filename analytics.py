# analytics.py v13.1
# Part of Project Genesis: Breathing Causality
# v13.1: HOTFIX - Corrected a NameError/UnboundLocalError in `generate_report`
#        by initializing variables before the conditional block.

import numpy as np
import matplotlib.pyplot as plt
import os
from termcolor import cprint

class GlobalAnalytics:
    """
    Analyzes the entire simulation run to extract emergent, global properties.
    """
    def __init__(self, num_points: int):
        cprint(f"4. Initializing Global Analytics Engine...", 'cyan', attrs=['bold'])
        if num_points <= 0:
            raise ValueError("Number of points must be positive.")
        self.num_points = num_points
        self.causality_flow_matrix = np.zeros((num_points, num_points), dtype=np.float32)
        self.entropy_history = []
        cprint(f"   -> Ready to analyze {num_points} nodes.", 'green')

    def analyze_step(self, psi: np.ndarray, causal_graph: list, frame_num: int):
        """Accumulates statistics from a single simulation step."""
        for i in range(self.num_points):
            for j in causal_graph[i]:
                self.causality_flow_matrix[j, i] += 1

        amplitudes_sq = np.abs(psi)**2
        non_zero_amps = amplitudes_sq[amplitudes_sq > 1e-12]
        shannon_entropy = -np.sum(non_zero_amps * np.log2(non_zero_amps))
        self.entropy_history.append(shannon_entropy)

    def generate_report(self, run_directory: str):
        """Generates and saves final plots and data files after the simulation."""
        cprint("\n--- Generating Global Analytics Report ---", 'yellow')
        report_dir = os.path.join(run_directory, 'analytics')
        os.makedirs(report_dir, exist_ok=True)

        # --- 1. Arrow of Time Analysis ---
        flow_matrix = self.causality_flow_matrix

        # --- FIX: Initialize variables before the conditional block ---
        total_flow_events = np.sum(flow_matrix)
        asymmetry_score = 0.0 # Default value if there's no flow

        if total_flow_events > 0:
            net_flow = np.sum(np.abs(flow_matrix - flow_matrix.T)) / 2
            # `total_flow_events` is the same as `total_flow` used in the formula
            asymmetry_score = net_flow / total_flow_events
            cprint(f"  -> Arrow of Time Asymmetry Score: {asymmetry_score:.4f}", 'cyan')
        else:
            cprint("  -> No causal flow events recorded.", 'cyan')

        # Save the raw flow matrix for deeper analysis
        flow_path = os.path.join(report_dir, 'causality_flow.npz')
        np.savez_compressed(flow_path, flow_matrix=flow_matrix, asymmetry_score=asymmetry_score)
        cprint(f"  -> Causal flow data saved to '{flow_path}'", 'cyan')

        # --- 2. Entropy Analysis (this part was already correct) ---
        if self.entropy_history:
            frames = np.arange(len(self.entropy_history))
            entropy = np.array(self.entropy_history)

            entropy_data_path = os.path.join(report_dir, 'entropy_history.npz')
            np.savez_compressed(entropy_data_path, frames=frames, entropy=entropy)
            cprint(f"  -> Entropy data saved to '{entropy_data_path}'", 'cyan')

            # Generate the plot
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(frames, entropy, color='#00FFFF', label='Global Shannon Entropy')
            ax.set_title("Evolution of Global Entropy", fontsize=16, pad=20)
            ax.set_xlabel("Simulation Step", fontsize=12)
            ax.set_ylabel("Entropy (bits)", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            entropy_plot_path = os.path.join(report_dir, 'entropy_evolution.png')
            fig.savefig(entropy_plot_path, dpi=150)
            plt.close(fig)
            cprint(f"  -> Entropy evolution plot saved to '{entropy_plot_path}'", 'cyan')

        cprint("--- Report Generation Complete ---", 'yellow')
