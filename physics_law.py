# physics_law.py v2.1
# Part of Project Chronos
# Author: Ben Carter, in consultation with the Genesis Team
# v2.1: "Corrected Action"
# - CRITICAL PHYSICS FIX: Corrected the sign in the `calculate_force` method
#   to properly represent the Schrödinger equation of motion (i*d/dt - H = 0).
# - Updated all gradients in `get_force_gradients` to reflect this change.
# - The class now correctly implements the physics for the relaxation engine.

import numpy as np
import scipy.sparse as sp

class DiscretizedDNLS:
    """
    Represents a parametrized physical law (DNLS) for the Relaxation paradigm.
    
    This class calculates the "force of inconsistency" for any given pair of
    states (psi_t, psi_t+1). The relaxation engine's goal is to find a history
    where this force is zero everywhere, satisfying the equation of motion.
    """
    def __init__(self, alpha: float, dt: float):
        """Initializes the physical law with its parameters."""
        if not isinstance(alpha, (int, float)) or not isinstance(dt, (int, float)):
            raise TypeError("alpha and dt must be numeric.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
            
        self.alpha = float(alpha)
        self.dt = float(dt)
        self.name = f"DNLS(alpha={self.alpha:.2f}, dt={self.dt:.4f})"

    def __repr__(self):
        """String representation for easy debugging."""
        return self.name

    def _get_hamiltonian_term(self, psi_t: np.ndarray, laplacian_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Calculates the action of the Hamiltonian H on a state psi_t.
        H*psi = (-Laplacian + V)*psi
        Note: The standard graph Laplacian is negative semi-definite, so -L is the
        correct kinetic term corresponding to -∇².
        """
        psi_t = psi_t.ravel()
        kinetic_term = -laplacian_matrix @ psi_t
        potential_term = self.alpha * (np.abs(psi_t)**2) * psi_t
        return kinetic_term + potential_term

    def calculate_force(self, psi_t: np.ndarray, psi_t_plus_1: np.ndarray, laplacian_matrix: sp.csr_matrix) -> np.ndarray:
        """
        Calculates the 'force of inconsistency' or 'tension' vector.
        This is the discrete version of the Schrödinger equation of motion:
        Force = i * d(psi)/dt - H(psi)*psi.
        The goal of the relaxation engine is to drive this Force vector to zero.
        
        Discrete form:
        F_t = i * (psi_{t+1} - psi_t)/dt - H(psi_t)*psi_t
        """
        psi_t = psi_t.ravel()
        psi_t_plus_1 = psi_t_plus_1.ravel()
        
        time_derivative_term = 1j * (psi_t_plus_1 - psi_t) / self.dt
        hamiltonian_term = self._get_hamiltonian_term(psi_t, laplacian_matrix)
        
        # --- THE FIX (v2.1) ---
        # The sign is corrected from '+' to '-' to match the EOM.
        force = time_derivative_term - hamiltonian_term
        return force

    def get_force_gradients(self, psi_t: np.ndarray, psi_t_plus_1: np.ndarray, laplacian_matrix: sp.csr_matrix) -> tuple:
        """
        Calculates the Wirtinger derivatives (Jacobians) of the `calculate_force` function
        w.r.t. all its complex inputs (psi_t, psi_t+1) and real parameters (alpha).

        Returns:
            tuple: ( (J_psi_t, K_psi_t), (J_psi_tp1, K_psi_tp1), grad_alpha )
        """
        psi_t = psi_t.ravel()
        N = len(psi_t)
        I = sp.identity(N, dtype=np.complex128, format='csr')

        # --- Gradient w.r.t. psi_{t+1} ---
        # F_t = (i/dt) * psi_{t+1} + ...
        # dF/d(psi_{t+1}) = i/dt * I
        grad_psi_t_plus_1_J = (1j / self.dt) * I
        # No conj(psi_{t+1}) term
        grad_psi_t_plus_1_K = sp.csr_matrix((N, N), dtype=np.complex128)

        # --- Gradient w.r.t. alpha ---
        # F_t = ... - alpha * |psi_t|^2 * psi_t
        # dF/d(alpha) = -|psi_t|^2 * psi_t
        grad_alpha = -(np.abs(psi_t)**2) * psi_t
        
        # --- Gradient w.r.t. psi_t ---
        # F_t = -(i/dt)*psi_t - (-L*psi_t + alpha*|psi_t|^2*psi_t)
        #     = -(i/dt)*psi_t + L*psi_t - alpha*psi_t*conj(psi_t)*psi_t
        
        # d/d(psi_t) of F_t gives the J component
        dF_dpsi_t_diag = -self.alpha * 2 * (np.abs(psi_t)**2)
        grad_psi_t_J = (-1j / self.dt) * I + laplacian_matrix + sp.diags(dF_dpsi_t_diag, format='csr')
        
        # d/d(conj(psi_t)) of F_t gives the K component
        dF_dpsi_conj_t_diag = -self.alpha * (psi_t**2)
        grad_psi_t_K = sp.diags(dF_dpsi_conj_t_diag, format='csr')

        return (grad_psi_t_J, grad_psi_t_K), (grad_psi_t_plus_1_J, grad_psi_t_plus_1_K), grad_alpha

    # Kept for generating ground truth data in tests.
    def forward_step_euler(self, psi_t: np.ndarray, laplacian_matrix: sp.csr_matrix) -> np.ndarray:
        """A simple Forward Euler integrator to generate test data."""
        hamiltonian_action = self._get_hamiltonian_term(psi_t, laplacian_matrix)
        # d(psi)/dt = -i*H*psi
        d_psi_dt = -1j * hamiltonian_action
        psi_t_plus_1 = psi_t + self.dt * d_psi_dt
        return psi_t_plus_1