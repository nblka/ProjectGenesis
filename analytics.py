# analytics.py v13.2
# Part of Project Genesis: Breathing Causality
# v13.2: "Complexity Metrics" - Added a new, more nuanced metric to
#        distinguish between different types of low-entropy states.
#        - Introduces the 'Concentration Index' (Coefficient of Variation of |psi|^2)
#          to measure the "clumpiness" of the energy distribution.
#        - This allows us to differentiate between a universe forming complex
#          structures and a universe decaying into a uniform, "dead" state.

import numpy as np
import matplotlib.pyplot as plt
import os
from termcolor import cprint

class GlobalAnalytics:
    """
    Analyzes the entire simulation run to extract emergent, global properties.
    """
    def __init__(self, num_points: int):
        cprint(f"4. Initializing Global Analytics Engine (v13.2)...", 'cyan', attrs=['bold'])
        if num_points <= 0:
            raise ValueError("Number of points must be positive.")
        self.num_points = num_points
        self.causality_flow_matrix = np.zeros((num_points, num_points), dtype=np.float32)
        self.entropy_history = []
        self.concentration_history = []
        self.global_max_amp_sq_so_far = 0.0

        cprint(f"   -> Ready to analyze {num_points} nodes.", 'green')

    def analyze_step(self, psi: np.ndarray, causal_graph: list, frame_num: int):
        """Accumulates statistics from a single simulation step."""
        # --- Causality flow accumulation (no changes) ---
        for i in range(self.num_points):
            for j in causal_graph[i]:
                self.causality_flow_matrix[j, i] += 1

        # --- Thermodynamics Analysis ---
        amplitudes_sq = np.abs(psi)**2

        current_max = np.max(amplitudes_sq)
        if current_max > self.global_max_amp_sq_so_far:
            self.global_max_amp_sq_so_far = current_max

        # 1. Shannon Entropy (as before)
        non_zero_amps = amplitudes_sq[amplitudes_sq > 1e-12]
        shannon_entropy = -np.sum(non_zero_amps * np.log2(non_zero_amps))
        self.entropy_history.append(shannon_entropy)

        # --- NEW: 2. Concentration Index (Structural Complexity) ---
        # We calculate the Coefficient of Variation for |psi|^2.
        # CV = Standard Deviation / Mean
        mean_amp_sq = np.mean(amplitudes_sq)
        if mean_amp_sq > 1e-12:
            std_dev_amp_sq = np.std(amplitudes_sq)
            concentration_index = std_dev_amp_sq / mean_amp_sq
        else:
            concentration_index = 0.0 # No variation if the field is zero

        self.concentration_history.append(concentration_index)

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

        # --- Thermodynamics and Complexity Analysis ---
        if self.entropy_history and self.concentration_history:
            frames = np.arange(len(self.entropy_history))
            entropy = np.array(self.entropy_history)
            concentration = np.array(self.concentration_history)

            # Save the raw data for both metrics
            thermo_data_path = os.path.join(report_dir, 'thermo_complexity_history.npz')
            np.savez_compressed(thermo_data_path, frames=frames, entropy=entropy, concentration=concentration)
            cprint(f"  -> Thermo & Complexity data saved to '{thermo_data_path}'", 'cyan')

            # --- NEW: Generate a combined plot with two Y-axes ---
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # Plot Entropy on the left Y-axis
            color_entropy = '#00FFFF' # Cyan
            ax1.set_xlabel("Simulation Step", fontsize=12)
            ax1.set_ylabel("Global Shannon Entropy (bits)", fontsize=12, color=color_entropy)
            ax1.plot(frames, entropy, color=color_entropy, label='Entropy (Order)')
            ax1.tick_params(axis='y', labelcolor=color_entropy)
            ax1.grid(True, linestyle='--', alpha=0.3)

            # Create a second Y-axis that shares the same X-axis
            ax2 = ax1.twinx()
            color_concentration = '#FFD700' # Gold
            ax2.set_ylabel("Concentration Index (Clumpiness)", fontsize=12, color=color_concentration)
            ax2.plot(frames, concentration, color=color_concentration, label='Concentration (Structure)')
            ax2.tick_params(axis='y', labelcolor=color_concentration)

            # Combine legends from both axes
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')

            fig.suptitle("Evolution of Order and Structure", fontsize=18, weight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle

            combo_plot_path = os.path.join(report_dir, 'thermo_complexity_evolution.png')
            fig.savefig(combo_plot_path, dpi=150)
            plt.close(fig)
            cprint(f"  -> Combined evolution plot saved to '{combo_plot_path}'", 'cyan')

        cprint("--- Report Generation Complete ---", 'yellow')
