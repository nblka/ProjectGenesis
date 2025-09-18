# styling.py v1.0
# Part of Project Genesis: Breathing Causality
# New in v16.0: "Centralized Styling"
# - Consolidates all color definitions and matplotlib styles into one place.
# - Allows for easy theme changes across the entire project.

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from termcolor import cprint

# --- Console Colors (using termcolor names) ---
# Usage: cprint("Hello", C.INFO)
class C:
    HEADER = 'magenta'
    SUBHEADER = 'cyan'
    SUCCESS = 'green'
    WARNING = 'yellow'
    ERROR = 'red'
    INFO = 'white'
    DEBUG = 'grey'
    BOLD_ATTR = ['bold']

# --- Matplotlib Plotting Styles ---

# Global style for all plots
plt.style.use('dark_background')

# Common font sizes
FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 12

# 1. Colors for Analytics Plots
COLOR_ENTROPY = '#00FFFF'      # Cyan
COLOR_CONCENTRATION = '#FFD700' # Gold

# 2. Colors and Colormaps for Frame Rendering
COLOR_BACKGROUND = '#08040E' # Very dark purple/black
COLOR_SUBSTRATE_EDGES = '#20182D' # Dim purple
COLOR_PARTICLE_HULL_FACE = '#FFD700' # Gold
COLOR_PARTICLE_HULL_EDGE = '#FFFFFF' # White

# 2a. Heatmap Colormap (Energy |psi|^2)
# Defines the gradient from deep space to a star-like core
_heatmap_colors = ["#08040E", "#0c0a2b", "#4a0b5e", "#9b1d5f", "#e2534b", "#fcae1e", "#f0f0c0"]
_heatmap_nodes  = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("genesis_heatmap", list(zip(_heatmap_nodes, _heatmap_colors)))

# 2b. Phase Colormap (for node colors)
# HSV is a good choice as it's cyclical, matching the 2*pi nature of phase.
PHASE_CMAP = plt.get_cmap('hsv')

# 2c. Causal Graph Link Colors
COLOR_CAUSAL_DARK = '#202020'  # Dark grey for the "past" part of the link
COLOR_CAUSAL_LIGHT = '#404040' # Lighter grey for the "future" part

# This allows for easy import and usage, e.g., `from styling import C, COLOR_ENTROPY`
if __name__ == "__main__":
    cprint("--- styling.py loaded ---", C.SUCCESS)
    cprint("This file contains centralized color and style constants.", C.INFO)
    cprint("Example usage:", C.SUBHEADER, attrs=C.BOLD_ATTR)
    cprint("  from styling import C, HEATMAP_CMAP", C.DEBUG)
    cprint("  cprint('Hello!', C.HEADER, attrs=C.BOLD_ATTR)", C.DEBUG)
