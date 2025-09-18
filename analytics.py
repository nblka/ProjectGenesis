# analytics.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "Style Refactoring"
# - All hardcoded colors and print styles are now imported from the
#   central `styling` module.
# - No changes to the physics or logic.

import numpy as np
import matplotlib.pyplot as plt
import os

# --- NEW: Import centralized styling ---
from styling import C, FONT_SIZE_LABEL, FONT_SIZE_TITLE, COLOR_ENTROPY, COLOR_CONCENTRATION
# We keep termcolor for the function, but use our constants for colors
from termcolor import cprint

class GlobalAnalytics:
    """
    Analyzes the entire simulation run to extract emergent, global properties.
    """
    def __init__(self, num_points: int):
        cprint(f"4. Initializing Global Analytics Engine (v16.0)...", C.SUBHEADER, attrs=C.BOLD_ATTR)
        if num_points <= 0:
            raise ValueError("Number of points must be positive.")
        self.num_points = num_points
        self.causality_flow_matrix = np.zeros((num_points, num_points), dtype=np.float32)
        self.entropy_history = []
        self.concentration_history = []
        self.global_interaction_source_max = 0.0

        cprint(f"   -> Ready to analyze {num_points} nodes.", C.SUCCESS)

    def analyze_step(self, interaction_source: np.ndarray, causal_graph: list, frame_num: int):
        """Accumulates statistics from a single simulation step."""
        # --- Causality flow accumulation (no changes) ---
        for i in range(self.num_points):
            if i < len(causal_graph) and causal_graph[i] is not None:
                for j in causal_graph[i]:
                    self.causality_flow_matrix[j, i] += 1

        # --- Thermodynamics Analysis (now uses interaction_source) ---
        current_max = np.max(interaction_source)
        if current_max > self.global_interaction_source_max:
            self.global_interaction_source_max = current_max

        # 1. Shannon Entropy
        # To be a valid probability distribution, the source must sum to 1.
        # We assume the source is |psi|^2 which is normalized.
        non_zero_source = interaction_source[interaction_source > 1e-12]
        shannon_entropy = -np.sum(non_zero_source * np.log2(non_zero_source))
        self.entropy_history.append(shannon_entropy)

        # 2. Concentration Index
        mean_source = np.mean(interaction_source)
        if mean_source > 1e-12:
            std_dev_source = np.std(interaction_source)
            concentration_index = std_dev_source / mean_source
        else:
            concentration_index = 0.0

        self.concentration_history.append(concentration_index)

    def generate_report(self, run_directory: str):
        """Generates and saves final plots and data files after the simulation."""
        cprint("\n--- Generating Global Analytics Report ---", C.WARNING)
        report_dir = os.path.join(run_directory, 'analytics')
        os.makedirs(report_dir, exist_ok=True)

        # --- 1. Arrow of Time Analysis ---
        flow_matrix = self.causality_flow_matrix
        total_flow_events = np.sum(flow_matrix)
        asymmetry_score = 0.0

        if total_flow_events > 0:
            net_flow = np.sum(np.abs(flow_matrix - flow_matrix.T)) / 2
            asymmetry_score = net_flow / total_flow_events
            cprint(f"  -> Arrow of Time Asymmetry Score: {asymmetry_score:.4f}", C.SUBHEADER)
        else:
            cprint("  -> No causal flow events recorded.", C.SUBHEADER)

        flow_path = os.path.join(report_dir, 'causality_flow.npz')
        np.savez_compressed(flow_path, flow_matrix=flow_matrix, asymmetry_score=asymmetry_score)
        cprint(f"  -> Causal flow data saved to '{flow_path}'", C.SUBHEADER)

        # --- Thermodynamics and Complexity Analysis ---
        if self.entropy_history and self.concentration_history:
            frames = np.arange(len(self.entropy_history))
            entropy = np.array(self.entropy_history)
            concentration = np.array(self.concentration_history)

            thermo_data_path = os.path.join(report_dir, 'thermo_complexity_history.npz')
            np.savez_compressed(thermo_data_path, frames=frames, entropy=entropy, concentration=concentration)
            cprint(f"  -> Thermo & Complexity data saved to '{thermo_data_path}'", C.SUBHEADER)

            # Generate a combined plot with two Y-axes
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # Plot Entropy
            ax1.set_xlabel("Simulation Step", fontsize=FONT_SIZE_LABEL)
            ax1.set_ylabel("Global Shannon Entropy (Order)", fontsize=FONT_SIZE_LABEL, color=COLOR_ENTROPY)
            ax1.plot(frames, entropy, color=COLOR_ENTROPY, label='Entropy (Order)')
            ax1.tick_params(axis='y', labelcolor=COLOR_ENTROPY)
            ax1.grid(True, linestyle='--', alpha=0.3)

            # Create second Y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel("Concentration Index (Structure)", fontsize=FONT_SIZE_LABEL, color=COLOR_CONCENTRATION)
            ax2.plot(frames, concentration, color=COLOR_CONCENTRATION, label='Concentration (Structure)')
            ax2.tick_params(axis='y', labelcolor=COLOR_CONCENTRATION)

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='best')

            fig.suptitle("Evolution of Order and Structure", fontsize=FONT_SIZE_TITLE, weight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            combo_plot_path = os.path.join(report_dir, 'thermo_complexity_evolution.png')
            fig.savefig(combo_plot_path, dpi=150)
            plt.close(fig)
            cprint(f"  -> Combined evolution plot saved to '{combo_plot_path}'", C.SUBHEADER)

        cprint("--- Report Generation Complete ---", C.WARNING)
