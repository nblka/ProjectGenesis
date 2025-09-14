# test_tracker.py v14.0
# FINAL VERSION: This test is now hermetic, robust, and self-consistent.
# - It uses the REAL `generate_crystal_topology` to create the substrate.
# - The MockSimulation is now just a minimal wrapper around a real TopologyData object.
# - The test stimulus (the particle) is now defined directly on the graph nodes,
#   removing any dependency on geometric shapes like Gaussians.
# - This version definitively tests the tracker's topological logic.

import unittest
import numpy as np
from termcolor import cprint

# --- Import the REAL components we are testing against ---
from topologies import TopologyData, generate_crystal_topology
from tracker import ParticleTracker

class MockSimulation:
    """
    A minimal, clean mock simulation environment. It holds a real TopologyData
    object and a psi field, and does nothing else.
    """
    def __init__(self, topology_data: TopologyData):
        self.substrate = topology_data
        self.psi = np.zeros(self.substrate.num_points, dtype=np.complex128)

    def set_psi_from_pattern(self, pattern: dict, background_amp: float = 0.01):
        """
        Sets the psi field based on a dictionary of {node_index: amplitude}.
        This is a purely topological way to define the stimulus.
        """
        amplitudes = np.full(self.substrate.num_points, background_amp)
        for node_idx, amp in pattern.items():
            amplitudes[node_idx] = amp

        phases = np.random.uniform(0, 2 * np.pi, self.substrate.num_points)
        self.psi = amplitudes * np.exp(1j * phases)


class TestParticleTrackerV14(unittest.TestCase):
    """A suite of tests for the final, topological ParticleTracker."""

    @classmethod
    def setUpClass(cls):
        """Set up a REAL topology once for all tests."""
        cprint("\n--- Setting up Test Environment for ParticleTracker v14.0 ---", 'yellow')
        # We use the real generator to ensure the neighbor list is 100% correct.
        cls.topology_data = generate_crystal_topology(width=20, height=20)

    def setUp(self):
        """Create a fresh mock_sim and tracker before each test."""
        self.mock_sim = MockSimulation(self.topology_data)
        self.tracker = ParticleTracker(
            stability_threshold=3,
            min_clump_size=4,      # Needs to be at least 4 for our test pattern
            ema_alpha=0.5,
            amp_threshold_factor=2.0
        )

    def test_full_lifecycle(self):
        """Tests the full particle lifecycle in a self-consistent environment."""
        cprint("  -> Testing full particle lifecycle on a real topology...", 'cyan')

        # --- Define our test particle PATTERN topologically ---
        # We will create a small, stable "plus" sign pattern and move it.
        # This removes any ambiguity from using geometric Gaussians.
        def get_pattern(center_idx):
            # A pattern of 5 nodes: center + 4 neighbors (up, down, left, right if available)
            neighbors = self.topology_data.neighbors[center_idx]
            pattern_nodes = {center_idx}
            if len(neighbors) >= 4:
                pattern_nodes.update(neighbors[:4])
            else: # For edge cases
                pattern_nodes.update(neighbors)
            return {node: 10.0 for node in pattern_nodes} # High amplitude for the pattern

        # --- Frames 0-1: Empty space ---
        self.mock_sim.set_psi_from_pattern({})
        self.tracker.analyze_frame(self.mock_sim, 0)
        self.assertEqual(len(self.tracker.tracked_particles), 0)

        # --- Frame 2: Particle is born ---
        cprint("    Frame 2: A particle (topological pattern) is born...")
        center_node_start = 50
        self.mock_sim.set_psi_from_pattern(get_pattern(center_node_start))
        self.tracker.analyze_frame(self.mock_sim, 2)
        self.assertEqual(len(self.tracker.tracked_particles), 1, "Particle should be born.")

        particle_id = list(self.tracker.tracked_particles.keys())[0]

        # --- Frames 3-5: Particle moves SMOOTHLY (topologically) ---
        center_node_current = center_node_start
        for i in range(1, 4):
            frame = 2 + i
            # Move to the first neighbor for simplicity
            center_node_current = self.topology_data.neighbors[center_node_current][0]
            cprint(f"    Frame {frame}: Moving pattern to center node {center_node_current}...")

            self.mock_sim.set_psi_from_pattern(get_pattern(center_node_current))
            stable_particles = self.tracker.analyze_frame(self.mock_sim, frame)

            # Check that the particle is not lost
            self.assertIn(particle_id, self.tracker.tracked_particles, f"Tracker lost particle on frame {frame}!")

            if frame == 5:
                p_data = self.tracker.tracked_particles[particle_id]
                self.assertEqual(p_data['age'], 4, "Particle age is incorrect.")
                self.assertEqual(p_data['state'], 'stable', "Particle should be stable.")
                self.assertEqual(len(stable_particles), 1, "Stable particle list should contain one particle.")

        # --- Frame 6: Disappears ---
        cprint("    Frame 6: Particle pattern is removed...")
        self.mock_sim.set_psi_from_pattern({})
        self.tracker.analyze_frame(self.mock_sim, 6)

        self.assertEqual(len(self.tracker.tracked_particles), 0, "Particle should be dead and removed.")

        cprint("Test Passed: Lifecycle handled correctly in a fully topological test.", 'green')


if __name__ == "__main__":
    unittest.main(verbosity=2)
