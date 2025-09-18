# visualization_strategies.py v1.1
# Part of Project Genesis: Breathing Causality
# v1.1: "Dependency Injection Fix"
# - The strategies are now stateless. They no longer assume access to global
#   colormaps.
# - `get_node_colors` now explicitly requires a `phase_cmap` argument, making
#   the dependency clear and the class more reusable and testable.

import numpy as np
import matplotlib.pyplot as plt

class AbstractVizStrategy:
    """Interface for all field visualization strategies."""
    def get_heatmap_values(self, field_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_node_colors(self, field_values: np.ndarray, phase_cmap: plt.Colormap) -> np.ndarray:
        """
        MODIFIED: Now requires the colormap to be passed in.
        """
        raise NotImplementedError

class ScalarFieldViz(AbstractVizStrategy):
    """Visualization strategy for a single-component ScalarField."""
    def get_heatmap_values(self, field_values: np.ndarray) -> np.ndarray:
        return np.abs(field_values.ravel())**2

    def get_node_colors(self, field_values: np.ndarray, phase_cmap: plt.Colormap) -> np.ndarray:
        """Node color is determined by the complex phase, using the provided colormap."""
        phases = np.angle(field_values.ravel())
        return phase_cmap((phases + np.pi) / (2 * np.pi))
