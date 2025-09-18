# field.py v16.0
# Part of Project Genesis: Breathing Causality
# New in v16.0: "The Field Abstraction"
# - Introduces an abstract base class `AbstractField` to define a common
#   interface for all physical fields in the simulation.
# - Implements `ScalarField` as the first concrete class, encapsulating
#   the behavior of our original single-component psi field.
# - Crucially, it separates the field's internal state (`values`) from
#   the quantity that drives causality (`get_interaction_source`), preparing
#   the architecture for more complex fields like spinors.

from abc import ABC, abstractmethod
import numpy as np
from termcolor import cprint

class AbstractField(ABC):
    """
    Abstract Base Class for all physical fields.

    This class defines the essential properties and methods that any field
    in the Project Genesis simulation must implement. It handles the storage
    of field values and enforces normalization.

    Attributes:
        values (np.ndarray): A complex numpy array of shape (num_points, num_components)
                             holding the state of the field at each substrate node.
        num_points (int): The number of nodes in the substrate.
        num_components (int): The number of components for this field type (e.g., 1 for scalar, 4 for Dirac spinor).
    """
    def __init__(self, num_points: int, num_components: int):
        if num_points < 0 or num_components <= 0:
            raise ValueError("Number of points and components must be positive.")
        self.num_points = num_points
        self.num_components = num_components

        # Initialize field values to a zero state.
        self.values = np.zeros((self.num_points, self.num_components), dtype=np.complex128)
        cprint(f"  > Field object created: {self.__class__.__name__} ({self.num_points} points, {self.num_components} components)", 'grey')

    def normalize(self):
        """
        Normalizes the total probability density of the field to 1.
        This ensures the conservation of the field's total "presence"
        across the simulation, as required by unitarity. The norm is calculated
        across all points and all components.
        """
        # np.linalg.norm works correctly on multi-dimensional arrays.
        norm = np.linalg.norm(self.values)
        if norm > 1e-12:
            self.values /= norm
        else:
            # A failsafe for a zero-field state to avoid division by zero.
            # This case should ideally not be reached in a physically meaningful simulation.
            pass

    @abstractmethod
    def get_interaction_source(self) -> np.ndarray:
        """
        Calculates and returns the scalar field that acts as the source for
        emergent causality. This is a critical abstraction, as different fields
        might influence causality in different ways.

        Returns:
            np.ndarray: A real-valued numpy array of shape (num_points,)
                        representing the "causal charge" at each node.
        """
        pass

class ScalarField(AbstractField):
    """
    A concrete implementation for a single-component complex scalar field (a boson field).
    This class represents the original `psi` field from previous versions.
    """
    def __init__(self, num_points: int):
        # A scalar field has exactly one component.
        super().__init__(num_points, num_components=1)

    def get_interaction_source(self) -> np.ndarray:
        """
        For a simple scalar field, the source of interaction (e.g., gravity)
        is its energy density, which is proportional to the squared amplitude |psi|^2.

        Returns:
            np.ndarray: A real-valued array of shape (num_points,) containing |psi|^2 at each node.
        """
        # .ravel() efficiently converts the (N, 1) `values` array to a flat (N,) array.
        return np.abs(self.values.ravel())**2

# --- Future Extension Point ---
# When we are ready to implement fermions, we will add the SpinorField class here.
# It will inherit from AbstractField and provide its own logic.
#
# class SpinorField(AbstractField):
#     def __init__(self, num_points: int):
#         # A Dirac spinor has 4 components.
#         super().__init__(num_points, num_components=4)
#
#     def get_interaction_source(self) -> np.ndarray:
#         # For a Dirac field, the source of gravity is not just |psi|^2.
#         # It would be a more complex Lorentz-invariant scalar, like psi_bar * psi.
#         # For now, we can placeholder it to return the total probability density.
#         # This is where the real physics of fermion interaction would be encoded.
#         return np.sum(np.abs(self.values)**2, axis=1)
#
# -----------------------------

# This block allows for independent testing of the module.
if __name__ == "__main__":
    cprint("\n--- Testing field.py v16.0 ---", 'yellow', attrs=['bold'])

    # Test 1: Create a ScalarField
    cprint("1. Creating a ScalarField for 100 points...", 'cyan')
    try:
        scalar_field = ScalarField(num_points=100)
        assert scalar_field.values.shape == (100, 1)
        cprint("  > SUCCESS: ScalarField created with correct shape.", 'green')
    except Exception as e:
        cprint(f"  > FAILED: {e}", 'red')

    # Test 2: Test normalization
    cprint("2. Testing normalization...", 'cyan')
    scalar_field.values[:] = 1.0 + 1.0j # Set all values to be non-normalized
    initial_norm_sq = np.sum(np.abs(scalar_field.values)**2)
    cprint(f"   - Initial |psi|^2 sum: {initial_norm_sq:.2f}", 'grey')
    scalar_field.normalize()
    final_norm_sq = np.sum(np.abs(scalar_field.values)**2)
    cprint(f"   - Final |psi|^2 sum: {final_norm_sq:.2f}", 'grey')
    assert np.isclose(final_norm_sq, 1.0)
    cprint("  > SUCCESS: Normalization works correctly.", 'green')

    # Test 3: Test interaction source
    cprint("3. Testing get_interaction_source...", 'cyan')
    # After normalization, each |psi|^2 should be 1.0 / 100
    expected_source_value = 1.0 / 100
    source = scalar_field.get_interaction_source()
    assert source.shape == (100,)
    assert np.allclose(source, expected_source_value)
    cprint("  > SUCCESS: Interaction source is calculated correctly as |psi|^2.", 'green')

    cprint("\n--- All tests for field.py passed! ---", 'yellow', attrs=['bold'])
