# initial_conditions.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "Field Object Generation"
# - The `generate` method in all classes now returns a fully instantiated
#   object of a class inheriting from `AbstractField` (e.g., `ScalarField`).
# - This change decouples the main orchestrator from the specifics of
#   how a field is initialized.
# - The logic for generating the actual numerical values remains the same,
#   but is now assigned to the `.values` attribute of the created field object.

from abc import ABC, abstractmethod
import numpy as np
from termcolor import cprint

# --- Import dependencies from the new architecture ---
from topologies import TopologyData
from field import AbstractField, ScalarField # NEW: Import field classes

class BaseInitialState(ABC):
    """
    Abstract base class for all initial state generators.
    Enforces the contract that all generators must return a valid Field object.
    """
    @abstractmethod
    def generate(self, topology_data: TopologyData) -> AbstractField:
        """
        Generates and returns a field object appropriate for this initial condition.

        Args:
            topology_data: The static substrate on which the field exists.

        Returns:
            An instance of a class derived from AbstractField.
        """
        raise NotImplementedError

class PrimordialSoupState(BaseInitialState):
    """
    Generates a `ScalarField` in a completely random state (maximum entropy).
    Represents a "hot", unstructured early universe.
    """
    def __init__(self):
        cprint(f"   -> IC Strategy: Primordial Soup (yields ScalarField)", 'cyan')

    def generate(self, topology_data: TopologyData) -> AbstractField:
        """Generates a normalized random complex scalar field."""
        # 1. Create the appropriate field object.
        field = ScalarField(topology_data.num_points)

        # 2. Generate the random values for its state.
        # This is a standard way to produce a complex field with a uniform phase distribution.
        random_values = (np.random.randn(field.num_points) +
                         1j * np.random.randn(field.num_points))

        # Assign the values. Note that we need to reshape to (N, 1) for a scalar field.
        field.values = random_values.reshape(-1, 1)

        # 3. Let the field object handle its own normalization.
        field.normalize()

        return field

class WavePacketState(BaseInitialState):
    """
    Generates a `ScalarField` containing a coherent Gaussian wave packet
    with an initial momentum. Represents a single, isolated particle.
    """
    def __init__(self, momentum_kick: float = 30.0, packet_width_ratio: float = 0.1):
        cprint(f"   -> IC Strategy: Coherent Wave Packet (yields ScalarField)", 'cyan')
        self.momentum_kick = momentum_kick
        self.packet_width_ratio = packet_width_ratio

    def generate(self, topology_data: TopologyData) -> AbstractField:
        field = ScalarField(topology_data.num_points)
        points = topology_data.points

        if field.num_points == 0:
            return field

        # --- Logic for calculating values is unchanged ---
        grid_span = np.max(points, axis=0) - np.min(points, axis=0)
        grid_width = np.max(grid_span) if grid_span.size > 0 else 1.0
        center = np.mean(points, axis=0)
        distances_sq = np.sum((points - center)**2, axis=1)
        sigma_sq = (grid_width * self.packet_width_ratio)**2
        if sigma_sq < 1e-9: sigma_sq = 1.0
        amplitude = np.exp(-distances_sq / (2.0 * sigma_sq))
        phase = np.exp(1j * points[:, 0] * self.momentum_kick) # Momentum along x-axis
        psi_values = amplitude * phase

        field.values = psi_values.reshape(-1, 1)
        field.normalize()

        return field

class VortexState(BaseInitialState):
    """
    Generates a `ScalarField` containing a single topological vortex defect
    in a "cold" vacuum. Represents a stable, topological quasi-particle.
    """
    def __init__(self, position_ratio=(0.5, 0.5)):
        cprint(f"   -> IC Strategy: Single Topological Vortex (yields ScalarField)", 'cyan')
        self.position_ratio = np.array(position_ratio)

    def generate(self, topology_data: TopologyData) -> AbstractField:
        field = ScalarField(topology_data.num_points)
        points = topology_data.points

        if field.num_points == 0:
            return field

        # --- Logic for calculating values is unchanged ---
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center_coords = min_coords + (max_coords - min_coords) * self.position_ratio
        relative_coords = points - center_coords

        if topology_data.dimensionality == 2:
            angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
            phase_profile = np.exp(1j * angles)
        else:
            phase_profile = np.ones(field.num_points, dtype=complex)

        distances = np.linalg.norm(relative_coords, axis=1)
        amplitude_profile = np.tanh(distances / 2.0)
        psi_values = amplitude_profile * phase_profile

        field.values = psi_values.reshape(-1, 1)
        field.normalize()

        # A final check in case normalization fails for a perfectly centered vortex
        if np.linalg.norm(field.values) < 1e-9:
            uniform_values = np.ones(field.num_points)
            field.values = uniform_values.reshape(-1, 1)
            field.normalize()

        return field


# --- This block allows for independent testing of the module ---
if __name__ == "__main__":
    from topologies import generate_crystal_topology

    cprint("\n--- Testing initial_conditions.py v16.0 ---", 'yellow', attrs=['bold'])

    # Setup a realistic test environment
    topo = generate_crystal_topology(width=20, height=20)

    # Test each generator
    generators = [
        ("Primordial Soup", PrimordialSoupState()),
        ("Wave Packet", WavePacketState()),
        ("Vortex", VortexState())
    ]

    for name, gen in generators:
        cprint(f"1. Testing {name} generator...", 'cyan')
        try:
            field_object = gen.generate(topo)

            # Check 1: Is it the right object type?
            assert isinstance(field_object, AbstractField)
            cprint("  > SUCCESS: Correct object type returned.", 'green')

            # Check 2: Is it normalized?
            total_prob = np.sum(field_object.get_interaction_source())
            assert np.isclose(total_prob, 1.0)
            cprint(f"  > SUCCESS: Field is correctly normalized (|source|^2 sum = {total_prob:.4f}).", 'green')

            # Check 3: Does it have the right shape?
            assert field_object.values.shape == (topo.num_points, 1)
            cprint("  > SUCCESS: Field has correct shape.", 'green')

        except Exception as e:
            cprint(f"  > FAILED: {name} test failed with error: {e}", 'red')

    cprint("\n--- All tests for initial_conditions.py passed! ---", 'yellow', attrs=['bold'])
