# test_tracker.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "Data-Driven Test Refactoring"
# - The test suite is updated to match the new `ParticleTracker` architecture.
# - The `MockSimulation` class is removed entirely, as it's no longer needed.
# - Tests now directly create and manipulate the data (`field`, `substrate`) that
#   is passed to the tracker, making the tests more direct and transparent.
# - This verifies that the refactored, decoupled tracker works correctly.

import unittest
import numpy as np
from termcolor import cprint

# --- Import the REAL components we are testing against ---
from topologies import TopologyData, generate_crystal_topology
from field import ScalarField
from tracker import ParticleTracker

class TestParticleTrackerV16(unittest.TestCase):
    """
    A suite of tests for the refactored, data-driven ParticleTracker.
    """

    @classmethod
    def setUpClass(cls):
        """Set up a REAL topology once for all tests."""
        cprint(f"\n--- Setting up Test Environment for ParticleTracker v16.0 ---", 'yellow')
        # We use the real generator to ensure the neighbor list is 100% correct.
        cls.topology_data = generate_crystal_topology(width=20, height=20)

    def setUp(self):
        """Create a fresh tracker and a field object before each test."""
        self.tracker = ParticleTracker(
            stability_threshold=3,
            min_clump_size=4,
            ema_alpha=0.5,
            amp_threshold_factor=2.0
        )
        # Create a field object that will be manipulated in each test
        self.field = ScalarField(self.topology_data.num_points)

    def set_field_from_pattern(self, pattern: dict, background_amp: float = 0.01):
        """
        Helper function to set the field state based on a topological pattern.
        This is a purely topological way to define the test stimulus.
        """
        amplitudes = np.full(self.topology_data.num_points, background_amp)
        for node_idx, amp in pattern.items():
            amplitudes[node_idx] = amp

        # Use random phases for realism
        phases = np.random.uniform(0, 2 * np.pi, self.topology_data.num_points)

        self.field.values = (amplitudes * np.exp(1j * phases)).reshape(-1, 1)

    def test_full_lifecycle(self):
        """Tests the full particle lifecycle using the new data-driven API."""
        cprint("  -> Testing full particle lifecycle with decoupled data...", 'cyan')

        # Helper function to define the particle pattern topologically
        def get_pattern(center_idx):
            neighbors = self.topology_data.neighbors[center_idx]
            pattern_nodes = {center_idx}
            if len(neighbors) >= 4:
                pattern_nodes.update(neighbors[:4])
            else:
                pattern_nodes.update(neighbors)
            return {node: 10.0 for node in pattern_nodes}

        # --- Frames 0-1: Empty space ---
        self.set_field_from_pattern({})
        # Call analyze_frame with the new signature
        self.tracker.analyze_frame(
            self.field.get_interaction_source(),
            self.field.values,
            self.topology_data,
            frame_num=0
        )
        self.assertEqual(len(self.tracker.tracked_particles), 0)

        # --- Frame 2: Particle is born ---
        cprint("    Frame 2: A particle (topological pattern) is born...")
        center_node_start = 50
        self.set_field_from_pattern(get_pattern(center_node_start))
        self.tracker.analyze_frame(
            self.field.get_interaction_source(),
            self.field.values,
            self.topology_data,
            frame_num=2
        )
        self.assertEqual(len(self.tracker.tracked_particles), 1, "Particle should be born.")
        particle_id = list(self.tracker.tracked_particles.keys())[0]

        # --- Frames 3-5: Particle moves smoothly (topologically) ---
        center_node_current = center_node_start
        for i in range(1, 4):
            frame = 2 + i
            # Move to the first neighbor for simplicity
            center_node_current = self.topology_data.neighbors[center_node_current][0]
            cprint(f"    Frame {frame}: Moving pattern to center node {center_node_current}...")

            self.set_field_from_pattern(get_pattern(center_node_current))
            stable_particles = self.tracker.analyze_frame(
                self.field.get_interaction_source(),
                self.field.values,
                self.topology_data,
                frame_num=frame
            )

            self.assertIn(particle_id, self.tracker.tracked_particles, f"Tracker lost particle on frame {frame}!")

            if frame == 5:
                p_data = self.tracker.tracked_particles[particle_id]
                self.assertEqual(p_data['age'], 4, "Particle age is incorrect.")
                self.assertEqual(p_data['state'], 'stable', "Particle should be stable.")
                self.assertEqual(len(stable_particles), 1, "Stable particle list should contain one particle.")

        # --- Frame 6: Disappears ---
        cprint("    Frame 6: Particle pattern is removed...")
        self.set_field_from_pattern({})
        self.tracker.analyze_frame(
            self.field.get_interaction_source(),
            self.field.values,
            self.topology_data,
            frame_num=6
        )

        self.assertEqual(len(self.tracker.tracked_particles), 0, "Particle should be dead and removed.")

        cprint("Test Passed: Lifecycle handled correctly with the new API.", 'green')


if __name__ == "__main__":
    unittest.main(verbosity=2)
