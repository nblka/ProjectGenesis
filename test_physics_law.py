# test_physics_law.py v2.0
# Part of Project Chronos

import unittest
import numpy as np
import scipy.sparse as sp
from termcolor import cprint

# Import the class to be tested
from physics_law import DiscretizedDNLS

class TestDiscretizedDNLS_V2(unittest.TestCase):
    """
    Unit tests for the DiscretizedDNLS v2.0 class.
    Focuses on verifying the correctness of the `calculate_force` method and its gradients.
    """
    def setUp(self):
        """Set up a small, predictable test environment before each test."""
        cprint(f"\n--- Running test: {self._testMethodName} ---", 'yellow')
        self.N = 4
        self.laplacian = sp.csr_matrix(np.array([
            [-2, 1, 0, 1], [1, -2, 1, 0], [0, 1, -2, 1], [1, 0, 1, -2]
        ]))
        self.psi_t = (np.arange(self.N) + 0.5 + 1j * (np.arange(self.N) + 0.5)).astype(np.complex128)
        self.psi_t_plus_1 = (np.arange(self.N) + 1.5 + 1j * (np.arange(self.N) - 0.5)).astype(np.complex128)

    def test_01_calculate_force_linear(self):
        """Tests the force calculation in the linear case (alpha=0)."""
        cprint("  -> Testing force calculation for alpha=0...", 'cyan')
        law = DiscretizedDNLS(alpha=0.0, dt=0.01)
        force = law.calculate_force(self.psi_t, self.psi_t_plus_1, self.laplacian)
        
        expected_time_deriv = 1j * (self.psi_t_plus_1 - self.psi_t) / law.dt
        expected_hamiltonian = -self.laplacian @ self.psi_t
        expected_force = expected_time_deriv - expected_hamiltonian
        
        self.assertTrue(np.allclose(force, expected_force))
        cprint("  -> SUCCESS: Linear force calculation is correct.", 'green')

    def test_02_force_is_zero_for_true_evolution(self):
        """Checks that the force is zero if psi_{t+1} comes from the forward step."""
        cprint("  -> Testing that F=0 for a consistent history step...", 'cyan')
        law = DiscretizedDNLS(alpha=2.5, dt=0.01)
        
        # Generate the "correct" next step
        psi_next_true = law.forward_step_euler(self.psi_t, self.laplacian)
        
        # Calculate force between psi_t and this correct next step
        force = law.calculate_force(self.psi_t, psi_next_true, self.laplacian)
        
        # Due to using Euler for forward step, it won't be exactly zero, but very small
        self.assertTrue(np.allclose(force, np.zeros(self.N), atol=1e-9))
        cprint("  -> SUCCESS: Force is effectively zero for a consistent step.", 'green')

    def test_03_gradient_force_numerical(self):
        """
        CRITICAL TEST: Verifies all analytical gradients of the force function
        by comparing them to numerical approximations.
        """
        cprint("  -> Numerically verifying ALL force gradients...", 'cyan')
        law = DiscretizedDNLS(alpha=-1.5, dt=0.01)
        epsilon = 1e-8
        
        # --- 1. Test gradients w.r.t. alpha ---
        (J_psi_t_an, K_psi_t_an), (J_psi_tp1_an, K_psi_tp1_an), grad_alpha_an = \
            law.get_force_gradients(self.psi_t, self.psi_t_plus_1, self.laplacian)
            
        law_plus = DiscretizedDNLS(alpha=law.alpha + epsilon, dt=law.dt)
        force_plus = law_plus.calculate_force(self.psi_t, self.psi_t_plus_1, self.laplacian)
        law_minus = DiscretizedDNLS(alpha=law.alpha - epsilon, dt=law.dt)
        force_minus = law_minus.calculate_force(self.psi_t, self.psi_t_plus_1, self.laplacian)
        grad_alpha_num = (force_plus - force_minus) / (2 * epsilon)
        self.assertTrue(np.allclose(grad_alpha_an, grad_alpha_num, atol=1e-6), "grad_alpha failed")
        cprint("    -> grad_alpha PASSED.", 'green')

        # --- 2. Test gradients w.r.t. psi_{t+1} (Wirtinger) ---
        v = np.random.randn(self.N) + 1j*np.random.randn(self.N)
        force_plus = law.calculate_force(self.psi_t, self.psi_t_plus_1 + epsilon*v, self.laplacian)
        force_minus = law.calculate_force(self.psi_t, self.psi_t_plus_1 - epsilon*v, self.laplacian)
        grad_psi_tp1_num_proj = (force_plus - force_minus) / (2 * epsilon)
        grad_psi_tp1_an_proj = J_psi_tp1_an @ v + K_psi_tp1_an @ v.conj()
        self.assertTrue(np.allclose(grad_psi_tp1_an_proj, grad_psi_tp1_num_proj, atol=1e-6), "grad_psi_{t+1} failed")
        cprint("    -> grad_psi_{t+1} PASSED.", 'green')
        
        # --- 3. Test gradients w.r.t. psi_t (Wirtinger) ---
        force_plus = law.calculate_force(self.psi_t + epsilon*v, self.psi_t_plus_1, self.laplacian)
        force_minus = law.calculate_force(self.psi_t - epsilon*v, self.psi_t_plus_1, self.laplacian)
        grad_psi_t_num_proj = (force_plus - force_minus) / (2 * epsilon)
        grad_psi_t_an_proj = J_psi_t_an @ v + K_psi_t_an @ v.conj()
        self.assertTrue(np.allclose(grad_psi_t_an_proj, grad_psi_t_num_proj, atol=1e-6), "grad_psi_t failed")
        cprint("    -> grad_psi_t PASSED.", 'green')
        cprint("  -> SUCCESS: All analytical gradients match numerical approximations.", 'green')

if __name__ == "__main__":
    unittest.main(verbosity=2)