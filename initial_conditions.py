# initial_conditions.py v13.1
# Part of Project Genesis: Breathing Causality
# v13.1: Final, robust version for the data-centric architecture.
# - All generators now accept a TopologyData object.
# - Vortex generation logic is improved to be more physically robust.

import numpy as np
from termcolor import cprint

# Import the data container class for type hinting
from topologies import TopologyData

class BaseInitialState:
    """Abstract base class for initial state generators."""
    def generate(self, topology_data: TopologyData) -> np.ndarray:
        raise NotImplementedError

class PrimordialSoupState(BaseInitialState):
    """Generates a completely random psi field (maximum entropy)."""
    def __init__(self):
        cprint(f"   -> IC Strategy: Primordial Soup", 'cyan')

    def generate(self, topology_data: TopologyData) -> np.ndarray:
        """Generates a normalized random complex vector."""
        num_points = topology_data.num_points

        # Generate random values for real and imaginary parts from a normal distribution.
        # This is a standard way to produce a complex field with uniform phase distribution.
        psi = (np.random.randn(num_points) + 1j * np.random.randn(num_points))

        norm = np.linalg.norm(psi)
        if norm > 1e-9:
            return psi / norm
        else:
            # Fallback for the astronomically unlikely case of a zero vector
            psi[0] = 1.0
            return psi

class WavePacketState(BaseInitialState):
    """Generates a coherent Gaussian wave packet with an initial momentum."""
    def __init__(self, momentum_kick: float = 30.0, packet_width_ratio: float = 0.1):
        cprint(f"   -> IC Strategy: Coherent Wave Packet (kick={momentum_kick})", 'cyan')
        self.momentum_kick = momentum_kick
        self.packet_width_ratio = packet_width_ratio

    def generate(self, topology_data: TopologyData) -> np.ndarray:
        points = topology_data.points

        if points.shape[0] == 0:
            return np.array([], dtype=np.complex128)

        # Determine the overall size of the topology for scaling the packet width
        grid_span = np.max(points, axis=0) - np.min(points, axis=0)
        # Use the largest dimension to define the characteristic width
        grid_width = np.max(grid_span) if grid_span.size > 0 else 1.0

        center = np.mean(points, axis=0)

        distances_sq = np.sum((points - center)**2, axis=1)
        sigma_sq = (grid_width * self.packet_width_ratio)**2
        if sigma_sq < 1e-9: sigma_sq = 1.0 # Avoid division by zero for tiny topologies

        amplitude = np.exp(-distances_sq / (2.0 * sigma_sq))
        phase = np.exp(1j * points[:, 0] * self.momentum_kick) # Momentum along x-axis

        psi = amplitude * phase
        return psi / np.linalg.norm(psi)

class VortexState(BaseInitialState):
    """Generates a single topological vortex defect in a cold vacuum."""
    def __init__(self, position_ratio=(0.5, 0.5)):
        cprint(f"   -> IC Strategy: Single Topological Vortex", 'cyan')
        self.position_ratio = np.array(position_ratio)

    def generate(self, topology_data: TopologyData) -> np.ndarray:
        points = topology_data.points
        num_points = topology_data.num_points

        if num_points == 0:
            return np.array([], dtype=np.complex128)

        # 1. Find the center for our vortex
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center_coords = min_coords + (max_coords - min_coords) * self.position_ratio

        # 2. Create the phase vortex
        # The phase of each point is its polar angle relative to the vortex center.
        relative_coords = points - center_coords
        # We only do this for 2D topologies. For 3D etc., a different logic is needed.
        if topology_data.dimensionality == 2:
            angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
            phase_profile = np.exp(1j * angles)
        else:
            # Fallback for other dimensions: no phase vortex
            phase_profile = np.ones(num_points, dtype=complex)

        # 3. Create the amplitude profile (a "hole" in the center)
        # We want the amplitude to be zero at the singularity and 1 far away.
        distances = np.linalg.norm(relative_coords, axis=1)
        # Use a tanh function for a smooth transition from 0 (at the center) to 1 (far away).
        # The divisor controls the "size" of the vortex core.
        amplitude_profile = np.tanh(distances / 2.0)

        psi = amplitude_profile * phase_profile

        # 4. Final normalization
        norm = np.linalg.norm(psi)
        if norm > 1e-9:
            return psi / norm
        else:
            # This can happen if all points are at the center, return a flat state
            return np.ones(num_points) / np.sqrt(num_points)
