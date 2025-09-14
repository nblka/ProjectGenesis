# test_analytics.py v13.0
# Unit tests for the GlobalAnalytics module.

import unittest
import numpy as np
import os
import shutil
from termcolor import cprint

# Import the component to be tested
from analytics import GlobalAnalytics

class TestGlobalAnalytics(unittest.TestCase):
    """A suite of tests for the global analytics engine."""

    def setUp(self):
        """Set up a fresh analytics object before each test."""
        cprint(f"\n--- Running test: {self._testMethodName} ---", 'yellow')
        self.num_points = 4
        self.analytics = GlobalAnalytics(self.num_points)
        self.test_dir = "test_run_dir_analytics"
        # Clean up any old test directories
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def tearDown(self):
        """Clean up the test directory after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_accumulation_and_reporting(self):
        """
        Tests the core logic of accumulating data and generating a report.
        """
        cprint("  -> Testing data accumulation and report generation...", 'cyan')

        # --- Frame 1 ---
        psi1 = np.array([0.1, 0.2, 0.9, 0.3], dtype=complex)
        psi1 /= np.linalg.norm(psi1) # Normalize
        # Causal graph (incoming): 0->1, 0->2, 0->3, 1->2, 1->3, 3->2
        # For simplicity, let's manually define a known causal graph
        causal_graph1 = [
            [],       # Nothing flows into 0
            [0],      # 0 -> 1
            [0, 1, 3],# 0->2, 1->2, 3->2
            [0, 1]    # 0->3, 1->3
        ]
        self.analytics.analyze_step(psi1, causal_graph1, 1)

        # --- Frame 2 ---
        psi2 = np.array([0.8, 0.1, 0.2, 0.4], dtype=complex)
        psi2 /= np.linalg.norm(psi2)
        # Causal graph (incoming): 1->0, 2->0, 2->1, 3->0, 3->2
        causal_graph2 = [
            [1, 2, 3], # 1->0, 2->0, 3->0
            [2],       # 2->1
            [3],       # 3->2
            []
        ]
        self.analytics.analyze_step(psi2, causal_graph2, 2)

        # --- Verification Step 1: Check internal state ---

        # 1a. Check entropy history
        self.assertEqual(len(self.analytics.entropy_history), 2, "Should have recorded entropy for 2 frames.")
        self.assertGreater(self.analytics.entropy_history[0], 0, "Entropy should be positive.")
        self.assertGreater(self.analytics.entropy_history[1], 0, "Entropy should be positive.")

        # 1b. Check causality flow matrix
        # Let's build the expected matrix manually
        # Frame 1 added: M[0,1]=1, M[0,2]=1, M[0,3]=1, M[1,2]=1, M[1,3]=1, M[3,2]=1
        # Frame 2 added: M[1,0]=1, M[2,0]=1, M[3,0]=1, M[2,1]=1, M[3,2]=1 (again)
        expected_matrix = np.array([
            [0., 1., 1., 1.], # Outgoing from 0
            [1., 0., 1., 1.], # Outgoing from 1
            [1., 1., 0., 0.], # Outgoing from 2
            [1., 0., 2., 0.]  # Outgoing from 3
        ], dtype=np.float32)

        # NOTE: My matrix is [sender, receiver]. So M[j,i] means j->i.
        # This is correct.

        self.assertTrue(
            np.array_equal(self.analytics.causality_flow_matrix, expected_matrix),
            "Accumulated causality matrix is incorrect."
        )
        cprint("  -> Internal state accumulated correctly.", 'green')

        # --- Verification Step 2: Check report generation ---
        self.analytics.generate_report(self.test_dir)

        # 2a. Check that files were created
        report_dir = os.path.join(self.test_dir, 'analytics')
        self.assertTrue(os.path.exists(report_dir), "Analytics directory was not created.")

        flow_path = os.path.join(report_dir, 'causality_flow.npz')
        entropy_plot_path = os.path.join(report_dir, 'entropy_evolution.png')
        entropy_data_path = os.path.join(report_dir, 'entropy_history.npz')

        self.assertTrue(os.path.exists(flow_path))
        self.assertTrue(os.path.exists(entropy_plot_path))
        self.assertTrue(os.path.exists(entropy_data_path))

        # 2b. Check content of saved data (with fixes)
        flow_path = os.path.join(self.test_dir, 'analytics', 'causality_flow.npz')
        entropy_data_path = os.path.join(self.test_dir, 'analytics', 'entropy_history.npz')

        with np.load(flow_path) as flow_data:
            self.assertTrue(np.array_equal(flow_data['flow_matrix'], expected_matrix))
            # --- FIX: Use the correct, machine-calculated value ---
            self.assertAlmostEqual(flow_data['asymmetry_score'], 3.0 / 11.0)

        with np.load(entropy_data_path) as entropy_data:
            self.assertEqual(len(entropy_data['entropy']), 2)
            self.assertTrue(np.allclose(entropy_data['entropy'], self.analytics.entropy_history))

        cprint("  -> Report and data files generated correctly.", 'green')
        cprint("Test Passed: GlobalAnalytics is robust and correct.", 'green')


if __name__ == "__main__":
    unittest.main(verbosity=2)
