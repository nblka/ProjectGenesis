# test_relaxation_engine.py v2.2
# Part of Project Chronos
# v2.2: "Correct Setup" - Fixes a critical AttributeError in the test setup.

import unittest
import numpy as np
import scipy.sparse as sp
from termcolor import cprint

# Import all necessary components
from physics_law import DiscretizedDNLS
from topologies import generate_crystal_topology
from adam_optimizer import AdamOptimizer
from relaxation_engine import RelaxationEngine

class TestRelaxationEngineV2_2(unittest.TestCase):
    """
    Final integration tests for the RelaxationEngine v2.1.
    """
    @classmethod
    def setUpClass(cls):
        """Generate a 'ground truth' history once for all tests."""
        cprint("\n--- Generating Ground Truth for Final Exam (v2.2) ---", 'yellow')
        cls.num_points = 10
        cls.num_steps = 20
        cls.topology = generate_crystal_topology(width=10, height=1)
        
        cls.true_law_linear = DiscretizedDNLS(alpha=0.0, dt=0.01)
        
        cls.ground_truth_history = np.zeros((cls.num_steps, cls.num_points), dtype=np.complex128)
        x = np.arange(cls.num_points)
        start_psi = np.exp(-(x - cls.num_points/3)**2 / 8) * np.exp(1j * x * 2.0)
        cls.ground_truth_history[0] = start_psi / np.linalg.norm(start_psi)

        # --- THE FIX IS HERE ---
        # Instead of calling a protected method with `None`, we create a temporary
        # engine instance just to get its correctly initialized laplacian matrix.
        # This is the proper object-oriented way.
        try:
            temp_engine = RelaxationEngine(
                start_state=cls.ground_truth_history[0],
                end_state_target=np.zeros_like(cls.ground_truth_history[0]), # end state doesn't matter here
                num_steps=cls.num_steps,
                initial_law=cls.true_law_linear,
                topology=cls.topology
            )
            laplacian = temp_engine.laplacian_matrix
        except Exception as e:
            # If this fails, the whole test suite should stop.
            raise RuntimeError(f"Failed to create a temporary engine for setup: {e}")

        # Now, generate the history using the true law and the correct laplacian
        for t in range(cls.num_steps - 1):
            # We use the Euler step for simplicity in ground truth generation,
            # as the relaxation engine must be robust enough to find the law anyway.
            cls.ground_truth_history[t+1] = cls.true_law_linear.forward_step_euler(
                cls.ground_truth_history[t], laplacian
            )
        
        cprint(f"  -> Ground truth generated successfully for {cls.num_steps} steps.", 'green')

    def test_01_operation_echo_finds_law_and_history(self):
        """
        The ultimate test: can the engine find the correct law and history?
        """
        cprint("\n--- Running Final Exam: Operation Echo ---", 'yellow')
        start_state = self.ground_truth_history[0]
        end_state = self.ground_truth_history[-1]
        
        initial_law_guess = DiscretizedDNLS(alpha=25.0, dt=self.true_law_linear.dt)
        cprint(f"  -> Starting with wrong law: {initial_law_guess.name}", 'cyan')

        engine = RelaxationEngine(start_state, end_state, self.num_steps, initial_law_guess, self.topology)
        
        optimal_law, optimal_history = engine.relax_and_learn(
            iterations=5000, 
            lr_law=0.1,
            lr_history=0.01,
            gradient_clip_threshold=1.0
        )

        self.assertFalse(np.isnan(optimal_history).any(), "NaNs detected!")
        self.assertFalse(np.isnan(optimal_law.alpha), "NaNs detected!")
        
        cprint(f"  -> True alpha = {self.true_law_linear.alpha}, Learned alpha = {optimal_law.alpha:.4f}", 'cyan')
        self.assertAlmostEqual(optimal_law.alpha, self.true_law_linear.alpha, places=2, 
                               msg="Engine failed to learn the correct physical law (alpha).")

        mse = np.mean(np.abs(optimal_history - self.ground_truth_history)**2)
        cprint(f"  -> Final MSE for history: {mse:.4e}", 'cyan')
        self.assertLess(mse, 1e-4, "Engine failed to reconstruct the correct history.")
        
        cprint("\n  ############################################", 'green')
        cprint("  #          OPERATION ECHO: SUCCESS         #", 'green', attrs=['bold'])
        cprint("  ############################################", 'green')
    
    def test_02_convergence_for_nonlinear_truth(self):
        """
        Bonus Test: Can the engine find a NON-LINEAR true law?
        """
        cprint("\n--- Running Bonus Test: Find Non-Linear Law ---", 'yellow')
        
        true_law_nonlinear = DiscretizedDNLS(alpha=-5.0, dt=0.01)
        
        gt_history_nl = np.zeros_like(self.ground_truth_history)
        gt_history_nl[0] = self.ground_truth_history[0]
        
        temp_engine = RelaxationEngine(gt_history_nl[0], gt_history_nl[-1], self.num_steps, true_law_nonlinear, self.topology)
        laplacian = temp_engine.laplacian_matrix

        for t in range(self.num_steps - 1):
            gt_history_nl[t+1] = true_law_nonlinear.forward_step_euler(gt_history_nl[t], laplacian)
            
        start_state = gt_history_nl[0]
        end_state = gt_history_nl[-1]
        
        initial_law_guess = DiscretizedDNLS(alpha=10.0, dt=true_law_nonlinear.dt)
        cprint(f"  -> True alpha = {true_law_nonlinear.alpha}. Starting guess = {initial_law_guess.alpha}", 'cyan')
        
        engine = RelaxationEngine(start_state, end_state, self.num_steps, initial_law_guess, self.topology)
        
        optimal_law, _ = engine.relax_and_learn(
            iterations=5000, 
            lr_law=0.1, 
            lr_history=0.01, 
            gradient_clip_threshold=1.0
        )
        
        cprint(f"  -> True alpha = {true_law_nonlinear.alpha}, Learned alpha = {optimal_law.alpha:.4f}", 'cyan')
        self.assertAlmostEqual(optimal_law.alpha, true_law_nonlinear.alpha, places=1,
                               msg="Engine failed to learn the correct NON-LINEAR law.")
        
        cprint("\n  -> SUCCESS: Engine correctly identified a non-linear universe.", 'green')

if __name__ == "__main__":
    unittest.main(verbosity=2)