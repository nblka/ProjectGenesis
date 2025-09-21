# adam_optimizer.py v1.0
# Part of Project Chronos
# Author: Ben Carter

import numpy as np

class AdamOptimizer:
    """
    A robust implementation of the Adam (Adaptive Moment Estimation) optimizer.
    
    This optimizer is well-suited for problems with noisy or sparse gradients.
    It adapts the learning rate for each parameter individually by keeping track
    of an exponentially decaying average of past gradients (1st moment, `m`) and
    past squared gradients (2nd moment, `v`).

    Reference: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initializes the optimizer with its hyperparameters.

        Args:
            learning_rate (float): The step size.
            beta1 (float): The exponential decay rate for the first moment estimates.
            beta2 (float): The exponential decay rate for the second-moment estimates.
            epsilon (float): A small constant for numerical stability (to prevent division by zero).
        """
        if not 0.0 < learning_rate: raise ValueError("Learning rate must be positive.")
        if not 0.0 <= beta1 < 1.0: raise ValueError("Beta1 must be in [0, 1).")
        if not 0.0 <= beta2 < 1.0: raise ValueError("Beta2 must be in [0, 1).")
        
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment vectors as None. They will be created with the
        # correct shape on the first call to update().
        self.m = None  # 1st moment vector (like momentum)
        self.v = None  # 2nd moment vector (like uncentered variance)
        self.t = 0     # Timestep counter for bias correction

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Computes the updated parameters given the current parameters and their gradients.

        Args:
            params (np.ndarray): The current values of the parameters to be updated.
            grads (np.ndarray): The gradients of the loss function with respect to the parameters.

        Returns:
            np.ndarray: The updated parameters.
        """
        # On the first call, initialize the moment vectors with the same shape as the parameters.
        if self.m is None:
            self.m = np.zeros_like(params, dtype=np.float64)
            self.v = np.zeros_like(params, dtype=np.float64)

        self.t += 1

        # Handle complex gradients correctly: moments are calculated for real and imag parts.
        # However, a simpler and often effective approach is to treat the complex gradient
        # as a vector in R^2n. For our use case, we will update moments with the complex
        # grads for `m` and with the squared magnitude for `v`.
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update biased second raw moment estimate. Note: uses squared magnitude of gradient.
        # This ensures `v` is always real and positive.
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.real(grads * grads.conj())
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # The main Adam update rule
        update_step = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Apply the update
        updated_params = params - update_step
        
        return updated_params