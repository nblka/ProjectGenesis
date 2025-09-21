# relaxation_engine.py v2.2
# Part of Project Chronos

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from termcolor import cprint

# Import our custom modules
from adam_optimizer import AdamOptimizer

class RelaxationEngine:
    def __init__(self, start_state, end_state_target, num_steps, initial_law, topology):
        self.num_points = len(start_state)
        self.start_state = start_state.copy().ravel()
        self.end_state_target = end_state_target.copy().ravel()
        self.num_steps = num_steps
        self.law = initial_law
        self.laplacian_matrix = self._build_laplacian(topology)
        self.history = np.zeros((num_steps, self.num_points), dtype=np.complex128)
        time_axis = np.linspace(0, 1, num_steps)
        for i in range(self.num_points):
            self.history[:, i] = np.interp(time_axis, [0, 1], [self.start_state[i], self.end_state_target[i]])
        noise_level = 0.1
        self.history[1:-1] += noise_level * (np.random.randn(*self.history[1:-1].shape) + 1j * np.random.randn(*self.history[1:-1].shape))
        self.history[0] = self.start_state
        self.history[-1] = self.end_state_target
        
    def _build_laplacian(self, topology):
        adj = np.zeros((self.num_points, self.num_points))
        degs = np.zeros(self.num_points)
        for i, n in enumerate(topology.neighbors):
            degs[i] = len(n); adj[i, n] = 1
        return sp.csr_matrix(np.diag(degs) - adj)

    def _calculate_gradients(self):
        grad_alpha_total = 0.0
        grad_history_total = np.zeros_like(self.history)
        total_loss = 0.0
        for t in range(self.num_steps - 1):
            psi_t = self.history[t]
            psi_t_plus_1 = self.history[t+1]
            force_t = self.law.calculate_force(psi_t, psi_t_plus_1, self.laplacian_matrix)
            total_loss += np.sum(np.abs(force_t)**2)
            (J_t, K_t), (J_tp1, K_tp1), grad_alpha = self.law.get_force_gradients(psi_t, psi_t_plus_1, self.laplacian_matrix)
            
            if t > 0:
                force_t_minus_1 = self.law.calculate_force(self.history[t-1], psi_t, self.laplacian_matrix)
                (J_tm1_tp1, K_tm1_tp1) = self.law.get_force_gradients(self.history[t-1], psi_t, self.laplacian_matrix)[1]
                grad_history_total[t] += (J_tm1_tp1.T @ force_t_minus_1.conj() + K_tm1_tp1.conj().T @ force_t_minus_1).conj()
            grad_history_total[t] += (J_t.T @ force_t.conj() + K_t.conj().T @ force_t).conj()
            grad_alpha_total += 2 * np.real(np.vdot(force_t, grad_alpha))

        mean_loss = total_loss / (self.num_steps - 1)
        return mean_loss, grad_alpha_total, grad_history_total
        
    def relax_and_learn(self, iterations: int, lr_law: float, lr_history: float,
                              gradient_clip_threshold: float = 10.0):
        cprint(f"Relaxing universe with initial law: {self.law.name}", "cyan")
        adam_law = AdamOptimizer(learning_rate=lr_law)
        adam_history = AdamOptimizer(learning_rate=lr_history)
        initial_alpha = self.law.alpha

        for i in tqdm(range(iterations), desc="Relaxing Universe", leave=False):
            loss, grad_alpha, grad_history = self._calculate_gradients()
            
            if np.isnan(loss) or np.isinf(loss):
                cprint(f"\nError: Loss is NaN/Inf at iteration {i}. Stopping.", "red"); break
            
            grad_norm = np.linalg.norm(grad_history)
            if grad_norm > gradient_clip_threshold:
                grad_history = grad_history / grad_norm * gradient_clip_threshold

            self.law.alpha = adam_law.update(self.law.alpha, grad_alpha)
            self.history[1:-1] = adam_history.update(self.history[1:-1], grad_history[1:-1])

            if i % (iterations // 10 or 1) == 0:
                tqdm.write(f"  Iter {i}/{iterations}: Loss={loss:.4e}, Alpha={self.law.alpha:.4f}")
        
        final_loss, _, _ = self._calculate_gradients()
        cprint(f"Relaxation complete. Final Loss (MSE of forces): {final_loss:.4e}", "green")
        cprint(f"Optimal law found: {self.law.name} (started from alpha={initial_alpha:.2f})", "green")
        return self.law, self.history