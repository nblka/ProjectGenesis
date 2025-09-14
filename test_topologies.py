# test_topologies.py v13.0
# Unit tests for the data-centric topology architecture.
# Verifies TopologyData, generator functions, and the TopologyFactory.

import unittest
import numpy as np
from termcolor import cprint

# Import the components we want to test
from topologies import TopologyData, generate_crystal_topology, TopologyFactory

class TestTopologyArchitecture(unittest.TestCase):
    """A suite of tests for the topology generation ecosystem."""

    def setUp(self):
        """This method is called before each test function."""
        cprint(f"\n--- Running test: {self._testMethodName} ---", 'yellow')

    def test_01_topology_data_container(self):
        """Tests the basic functionality and validation of the TopologyData container."""
        cprint("  -> Testing TopologyData class...", 'cyan')

        # Test successful creation
        points = np.array([[0,0], [1,1]])
        neighbors = [[1], [0]]
        topo = TopologyData(points=points, neighbors=neighbors, dimensionality=2)
        self.assertEqual(topo.num_points, 2)
        self.assertEqual(topo.dimensionality, 2)

        # Test validation: mismatched lengths
        with self.assertRaises(AssertionError):
            cprint("  -> Testing validation: mismatched points and neighbors...", 'cyan')
            TopologyData(points=points, neighbors=[[1]], dimensionality=2) # Only one neighbor list

        # Test validation: mismatched dimensionality
        with self.assertRaises(AssertionError):
            cprint("  -> Testing validation: mismatched points and dimensionality...", 'cyan')
            TopologyData(points=points, neighbors=neighbors, dimensionality=3) # Points are 2D

        cprint("Test Passed: TopologyData container is robust.", 'green')

    def test_02_crystal_topology_generator(self):
        """Tests the crystal topology generator for correctness."""
        cprint("  -> Testing generate_crystal_topology function...", 'cyan')

        # Use a small, predictable grid size
        width, height = 4, 4
        topo_data = generate_crystal_topology(width=width, height=height)

        # --- Verification ---
        # 1. Check basic properties
        self.assertIsInstance(topo_data, TopologyData)
        self.assertEqual(topo_data.num_points, width * height)
        self.assertEqual(topo_data.dimensionality, 2)

        # 2. Check a known property of the hexagonal lattice
        # The total number of edges in a grid like this can be calculated.
        # Edges = (width-1)*height + (height-1)*width + (height-1)*(width-1)
        # For a 4x4 grid: 3*4 + 3*4 + 3*3 = 12 + 12 + 9 = 33
        total_edges = sum(len(n) for n in topo_data.neighbors) // 2
        self.assertEqual(total_edges, 33)

        # 3. Check a central node for 6 neighbors
        # Index for a central node in a 4x4 grid (e.g., node 5)
        center_node_idx = 1 * width + 1
        self.assertEqual(len(topo_data.neighbors[center_node_idx]), 6)

        cprint("Test Passed: Crystal generator creates a valid lattice.", 'green')

    def test_03_topology_factory(self):
        """Tests the TopologyFactory's ability to create correct objects."""
        cprint("  -> Testing TopologyFactory...", 'cyan')

        # 1. Test creating a crystal
        params_crystal = {'width': 5, 'height': 5}
        topo_crystal = TopologyFactory.create('crystal', params_crystal)
        self.assertIsInstance(topo_crystal, TopologyData)
        self.assertEqual(topo_crystal.num_points, 25)
        self.assertEqual(topo_crystal.dimensionality, 2)

        # 2. Test default parameters
        topo_default = TopologyFactory.create('crystal', {}) # Empty params
        self.assertIsInstance(topo_default, TopologyData)
        self.assertEqual(topo_default.num_points, 80 * 60) # Should use defaults

        # 3. Test for raising an error on unknown type
        with self.assertRaises(ValueError):
            cprint("  -> Testing validation: unknown topology type...", 'cyan')
            TopologyFactory.create('non_existent_topology', {})

        cprint("Test Passed: TopologyFactory works as expected.", 'green')

# --- This allows running the tests directly from the command line ---
if __name__ == "__main__":
    unittest.main(verbosity=2)
