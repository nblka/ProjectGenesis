# topologies.py v13.1
# Part of Project Genesis: Breathing Causality
# v13.1: Final, robust data-centric architecture.
# - Contains the passive TopologyData container.
# - Contains generator functions for creating topologies.
# - Contains a factory to select the correct generator.

import numpy as np
from termcolor import cprint

class TopologyData:
    """A simple, passive data container for the static substrate."""
    def __init__(self, points: np.ndarray, neighbors: list, dimensionality: int, **kwargs):
        self.points = points
        self.neighbors = neighbors
        self.dimensionality = dimensionality
        self.num_points = len(points)

        # Store any additional metadata (like width, height for crystals)
        self.metadata = kwargs
        self.__dict__.update(kwargs) # Allow direct attribute access, e.g., topology_data.width

        # Validation
        assert len(points) == len(neighbors), "Points and neighbors lists must have the same length."
        if points.ndim == 2: # Ensure this check doesn't fail for empty arrays
            assert points.shape[1] == dimensionality, "Points' dimension must match the specified dimensionality."

# --- GENERATOR FUNCTIONS ---

def generate_crystal_topology(width: int = 80, height: int = 60) -> TopologyData:
    """Generator function for a 2D crystal topology."""
    cprint(f"1. Generating Substrate: 2D Crystal ({width}x{height})", 'cyan', attrs=['bold'])
    num_points = width * height
    points = np.zeros((num_points, 2), dtype=float)
    neighbors = [[] for _ in range(num_points)]

    for r in range(height):
        for q in range(width):
            idx = r * width + q
            px = q + 0.5 * (r % 2); py = r * np.sqrt(3) / 2
            points[idx] = [px, py]

            if q > 0:
                left_idx = idx - 1
                neighbors[idx].append(left_idx)
                neighbors[left_idx].append(idx)
            if r > 0:
                if r % 2 == 0:
                    if q > 0:
                        top_left_idx = idx - width - 1
                        neighbors[idx].append(top_left_idx)
                        neighbors[top_left_idx].append(idx)
                    top_right_idx = idx - width
                    neighbors[idx].append(top_right_idx)
                    neighbors[top_right_idx].append(idx)
                else:
                    top_left_idx = idx - width
                    neighbors[idx].append(top_left_idx)
                    neighbors[top_left_idx].append(idx)
                    if q < width - 1:
                        top_right_idx = idx - width + 1
                        neighbors[idx].append(top_right_idx)
                        neighbors[top_right_idx].append(idx)

    points -= np.mean(points, axis=0)

    return TopologyData(points=points, neighbors=neighbors, dimensionality=2, width=width, height=height)

# --- FACTORY ---

class TopologyFactory:
    @staticmethod
    def create(topology_type: str, params: dict) -> TopologyData:
        if topology_type == 'crystal':
            return generate_crystal_topology(
                width=params.get('width', 80),
                height=params.get('height', 60)
            )
        # Add other generators here in the future
        # elif topology_type == 'random_3d':
        #     ...
        else:
            raise ValueError(f"Unknown topology type: '{topology_type}'")
