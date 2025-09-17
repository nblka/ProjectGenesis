# test_simulation.py v15.1
# Integration tests for the Simulation core (Paradigm Shift version).
# This test verifies the physical behavior of the simulation engine.

import unittest
import numpy as np
from termcolor import cprint

# --- Import all REAL components for the integration test ---
from topologies import TopologyData, generate_crystal_topology
from causality import AmplitudeConvergentCausality, AmplitudeDivergentCausality
from initial_conditions import BaseInitialState
from simulation import Simulation

# --- Mock Initial Condition for precise testing ---
class HotSpotState(BaseInitialState):
    """A generator that places all amplitude on a single node."""
    def __init__(self, spot_index=0):
        self.spot_index = spot_index
    def generate(self, topology_data: TopologyData) -> np.ndarray:
        psi = np.zeros(topology_data.num_points, dtype=complex)
        psi[self.spot_index] = 1.0 + 0.0j
        return psi

class FlatState(BaseInitialState):
    """A generator for a completely flat, uniform amplitude state."""
    def generate(self, topology_data: TopologyData) -> np.ndarray:
        amp = 1.0 / np.sqrt(topology_data.num_points)
        return np.full(topology_data.num_points, amp, dtype=complex)

class IslandState(BaseInitialState):
    def __init__(self, island_nodes: list):
        self.island_nodes = island_nodes
    def generate(self, topology_data: TopologyData) -> np.ndarray:
        psi = np.zeros(topology_data.num_points, dtype=complex)
        amp = 1.0 / np.sqrt(len(self.island_nodes))
        psi[self.island_nodes] = amp
        return psi

class TestSimulationV15(unittest.TestCase):
    """A suite of tests for the physically corrected Simulation class."""

    @classmethod
    def setUpClass(cls):
        """Set up a single, simple topology for all tests."""
        cprint(f"\n--- Setting up Test Environment for Simulation v15.1 ---", 'yellow')
        # A 3x3 grid is small enough to reason about, but not trivial.
        cls.topology_data = generate_crystal_topology(width=3, height=3)
        cls.causality_gen = AmplitudeConvergentCausality()

    def test_01_probability_conservation(self):
        """Test 1: Ensures |psi|^2 = 1 is maintained over many steps."""
        cprint("  -> Testing probability conservation...", 'cyan')
        initial_state_gen = HotSpotState(spot_index=4) # Center node
        sim = Simulation(self.topology_data, self.causality_gen, initial_state_gen)

        for _ in range(50):
            sim.update_step()
            self.assertTrue(
                np.isclose(np.sum(np.abs(sim.psi)**2), 1.0),
                f"Normalization failed! |psi|^2 = {np.sum(np.abs(sim.psi)**2)}"
            )
        sim.close()
        cprint("Test Passed: Probability is conserved.", 'green')

    def test_02_flat_state_stability(self):
        """Test 2: A uniform state should not evolve in amplitude."""
        cprint("  -> Testing stability of a flat (zero-Laplacian) state...", 'cyan')
        initial_state_gen = FlatState()
        sim = Simulation(self.topology_data, self.causality_gen, initial_state_gen)

        initial_amps_sq = np.abs(sim.psi)**2
        sim.update_step()
        final_amps_sq = np.abs(sim.psi)**2

        # Amplitudes should not change, only phases might.
        self.assertTrue(
            np.allclose(initial_amps_sq, final_amps_sq),
            "Amplitudes of a flat state should not evolve."
        )
        sim.close()
        cprint("Test Passed: Flat state is stable.", 'green')

    def test_03_island_evolution_divergent(self):
        """Test 3a: An 'island' must spread out under Divergent causality."""
        cprint("  -> Testing island evolution (Divergent flow)...", 'cyan')

        center_node = 4
        island_nodes = [center_node] + self.topology_data.neighbors[center_node]

        initial_state_gen = IslandState(island_nodes=island_nodes)
        causality_gen = AmplitudeDivergentCausality()
        sim = Simulation(self.topology_data, causality_gen, initial_state_gen)

        energy_before = np.sum(np.abs(sim.psi[island_nodes])**2)
        self.assertAlmostEqual(energy_before, 1.0)

        sim.update_step()

        energy_after = np.sum(np.abs(sim.psi[island_nodes])**2)

        self.assertLess(
            energy_after,
            energy_before,
            "Energy did not flow outwards from the island under Divergent rule."
        )
        sim.close()
        cprint("Test Passed: Island correctly diffuses (Divergent).", 'green')


    def test_04_causal_graph_correctness(self):
        """Test 4: The returned causal graph must match the state of psi AFTER evolution."""
        cprint("  -> Testing correctness of the returned causal graph...", 'cyan')
        initial_state_gen = HotSpotState(spot_index=4)
        sim = Simulation(self.topology_data, self.causality_gen, initial_state_gen)

        # Run one step to get the state psi_after and the returned graph
        returned_causal_graph = sim.update_step()
        psi_after = sim.psi

        # Now, manually calculate what the causal graph should be for psi_after
        expected_causal_graph = self.causality_gen.evolve(
            psi_after,
            self.topology_data.neighbors,
            self.topology_data.num_points
        )

        self.assertEqual(
            returned_causal_graph,
            expected_causal_graph,
            "Returned causal graph does not match the final state of psi."
        )
        sim.close()
        cprint("Test Passed: Causal graph is correctly generated from the new psi state.", 'green')

    def test_05_island_evolution_convergent(self):
        """Test 3b: An 'island' must contract or stabilize under Convergent causality."""
        cprint("  -> Testing island evolution (Convergent flow)...", 'cyan')

        center_node = 4
        island_nodes = [center_node] + self.topology_data.neighbors[center_node]

        initial_state_gen = IslandState(island_nodes=island_nodes)
        causality_gen = AmplitudeConvergentCausality()
        sim = Simulation(self.topology_data, causality_gen, initial_state_gen)

        energy_before = np.sum(np.abs(sim.psi[island_nodes])**2)
        sim.update_step()
        energy_after = np.sum(np.abs(sim.psi[island_nodes])**2)

        self.assertGreaterEqual(
            energy_after,
            energy_before * 0.99, # Даем небольшой допуск на численные эффекты
            "Energy should not significantly dissipate from the island under Convergent rule."
        )
        sim.close()
        cprint("Test Passed: Island correctly contracts/stabilizes (Convergent).", 'green')

if __name__ == "__main__":
    unittest.main(verbosity=2)
