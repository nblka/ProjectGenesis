# test_initial_conditions.py v13.0
# Unit tests for the data-centric initial condition generators.

import unittest
import numpy as np
from termcolor import cprint

# Import the components to be tested
from topologies import TopologyData, generate_crystal_topology
from initial_conditions import BaseInitialState, PrimordialSoupState, WavePacketState, VortexState

class TestInitialConditions(unittest.TestCase):
    """A suite of tests for the initial state generator classes."""

    def setUp(self):
        """Set up a mock topology_data object to be used by all tests."""
        cprint(f"\n--- Running test: {self._testMethodName} ---", 'yellow')
        # We use a real generator to create a realistic test environment
        self.topology_data = generate_crystal_topology(width=20, height=20)
        self.num_points = self.topology_data.num_points

    def _run_common_tests(self, generator: BaseInitialState, generator_name: str):
        """A helper function to run the same set of checks for any generator."""
        cprint(f"  -> Testing generator: {generator_name}...", 'cyan')

        # 1. Generate the psi field
        psi = generator.generate(self.topology_data)

        # 2. Check the type and shape
        self.assertIsInstance(psi, np.ndarray, f"{generator_name} should return a numpy array.")
        self.assertEqual(psi.shape, (self.num_points,), f"{generator_name} returned an array of incorrect shape.")
        self.assertTrue(np.iscomplexobj(psi), f"{generator_name} should return a complex-valued array.")

        # 3. CRITICAL: Check for normalization
        total_probability = np.sum(np.abs(psi)**2)
        self.assertTrue(
            np.isclose(total_probability, 1.0),
            f"{generator_name} did not return a normalized vector! |psi|^2 = {total_probability}"
        )

        cprint(f"  -> Test Passed for {generator_name}.", 'green')

    def test_01_primordial_soup(self):
        """Tests the PrimordialSoupState generator."""
        generator = PrimordialSoupState()
        self._run_common_tests(generator, "PrimordialSoupState")

        # Specific test for soup: check that it's not a zero or constant vector
        psi = generator.generate(self.topology_data)
        self.assertFalse(np.all(psi == 0), "Soup state should not be a zero vector.")
        self.assertFalse(len(np.unique(psi)) < 5, "Soup state should be highly random.")

    def test_02_wave_packet(self):
        """Tests the WavePacketState generator."""
        generator = WavePacketState()
        self._run_common_tests(generator, "WavePacketState")

        # Specific test for wave packet: check that the amplitude is peaked at the center
        psi = generator.generate(self.topology_data)
        amplitudes_sq = np.abs(psi)**2
        center_idx = self.num_points // 2 + 10 # Approximate center of the grid

        # The amplitude at the center should be significantly higher than at the edges
        self.assertGreater(
            amplitudes_sq[center_idx],
            amplitudes_sq[0] * 10,
            "Wave packet amplitude should be peaked at the center."
        )

    def test_03_vortex(self):
        """Tests the VortexState generator."""
        generator = VortexState()
        self._run_common_tests(generator, "VortexState")

        # Specific test for vortex: check for the topological charge
        psi = generator.generate(self.topology_data)

        # Find the center (should be the point with the minimum amplitude)
        center_idx = np.argmin(np.abs(psi))
        center_neighbors = self.topology_data.neighbors[center_idx]

        # Calculate the total phase change when traversing the neighbors
        neighbor_phases = np.angle(psi[center_neighbors])

        # To calculate charge, we need to get angles relative to the center node
        center_point = self.topology_data.points[center_idx]
        neighbor_points = self.topology_data.points[center_neighbors]
        geometric_angles = np.arctan2(neighbor_points[:,1] - center_point[1], neighbor_points[:,0] - center_point[0])

        # Sort neighbors by their geometric angle
        sorted_indices = np.argsort(geometric_angles)
        sorted_neighbor_phases = neighbor_phases[sorted_indices]

        # Calculate phase differences between sorted neighbors
        phase_diffs = np.diff(sorted_neighbor_phases)
        # Handle the wrap-around from 2pi to -2pi
        phase_diffs = np.angle(np.exp(1j * phase_diffs))

        total_phase_change = np.sum(phase_diffs)
        # Add the final jump from the last neighbor back to the first
        final_jump = sorted_neighbor_phases[0] - sorted_neighbor_phases[-1]
        total_phase_change += np.angle(np.exp(1j * final_jump))

        topological_charge = total_phase_change / (2 * np.pi)

        cprint(f"  -> Calculated topological charge: {topological_charge:.4f}", 'cyan')
        self.assertTrue(
            np.isclose(topological_charge, 1.0),
            f"Vortex state should have a topological charge of ~1.0, but got {topological_charge}"
        )

# --- This allows running the tests directly from the command line ---
if __name__ == "__main__":
    unittest.main(verbosity=2)
