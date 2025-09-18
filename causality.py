# causality.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "Source Abstraction"
# - The `evolve` method in all causality strategies is refactored to accept
#   a generic, real-valued `interaction_source` numpy array instead of
#   the complex `psi` field.
# - This completely decouples the causality logic from the underlying physics
#   of the field. The module no longer cares if the source is |psi|^2 from a
#   scalar field or psi_bar*psi from a spinor field.
# - The name of the module is now more accurate: it doesn't "evolve" anything,
#   it "computes" the emergent graph. The classes are renamed for clarity.

import numpy as np
from abc import ABC, abstractmethod
from termcolor import cprint

class AbstractCausalityComputer(ABC):
    """
    Abstract Base Class for all emergent causality strategies.

    Defines the interface for modules that compute the instantaneous
    directed causal graph G_causal(t) from a scalar interaction source field
    and the static undirected substrate.
    """
    def __init__(self):
        self.strategy_name = "Abstract"

    @abstractmethod
    def compute(self, interaction_source: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        """
        Computes the INCOMING directed adjacency list for the current frame.

        Args:
            interaction_source (np.ndarray): The real-valued scalar field that
                                             drives causality (e.g., energy density).
            undirected_neighbors (list): The static substrate's undirected adjacency list.
            num_points (int): The total number of points in the substrate.

        Returns:
            list: An INCOMING directed adjacency list, where incoming_neighbors[i]
                  contains a list of nodes `j` such that the emergent causal
                  edge is defined as j -> i.
        """
        pass

class ConvergentCausality(AbstractCausalityComputer):
    """
    HYPOTHESIS 1: "Convergent Flow" or "Uphill Causality"

    Causality flows from lower source intensity to higher intensity regions.
    This can be interpreted as information concentrating towards areas of high
    "presence" or mass-energy, akin to a gravitational pull.
    """
    def __init__(self):
        super().__init__()
        self.strategy_name = "Convergent (Uphill)"
        cprint(f"   -> Causality Strategy: {self.strategy_name}", 'cyan')

    def compute(self, interaction_source: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        """
        Generates an incoming directed graph based on the rule:
        An edge j -> i exists if the source intensity at i is greater than at j.
        """
        incoming_neighbors = [[] for _ in range(num_points)]

        # The logic is now cleaner as it operates directly on the source.
        for i in range(num_points):
            source_i = interaction_source[i]
            for j in undirected_neighbors[i]:
                # To avoid double-counting, we only process pairs where i > j
                if i > j:
                    source_j = interaction_source[j]

                    if source_i > source_j:
                        # Source at i is higher, so flow is j -> i.
                        incoming_neighbors[i].append(j)
                    elif source_j > source_i:
                        # Source at j is higher, so flow is i -> j.
                        incoming_neighbors[j].append(i)
                    # If sources are equal, no causal edge is formed.

        return incoming_neighbors

class DivergentCausality(AbstractCausalityComputer):
    """
    HYPOTHESIS 2: "Divergent Flow" or "Downhill Causality"

    Causality flows from higher source intensity to lower intensity regions.
    This can be interpreted as presence/energy dissipating or flowing outwards,
    like heat or a radiating source.
    """
    def __init__(self):
        super().__init__()
        self.strategy_name = "Divergent (Downhill)"
        cprint(f"   -> Causality Strategy: {self.strategy_name}", 'cyan')

    def compute(self, interaction_source: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        incoming_neighbors = [[] for _ in range(num_points)]

        for i in range(num_points):
            source_i = interaction_source[i]
            for j in undirected_neighbors[i]:
                if i > j:
                    source_j = interaction_source[j]

                    if source_i > source_j:
                        # Source at i is higher, so flow is i -> j.
                        incoming_neighbors[j].append(i)
                    elif source_j > source_i:
                        # Source at j is higher, so flow is j -> i.
                        incoming_neighbors[i].append(j)

        return incoming_neighbors

# --- Example Usage for Testing ---
if __name__ == "__main__":
    cprint("\n--- Testing causality.py v16.0 ---", 'yellow', attrs=['bold'])

    # Test Setup
    mock_neighbors = [[1, 3], [0, 2], [1, 3], [0, 2]]
    mock_num_points = 4
    # The module now takes the interaction source directly.
    mock_source = np.array([1.0, 4.0, 16.0, 9.0])
    cprint(f"Test substrate: square graph. Interaction source: {mock_source}", 'white')

    # Test 1: Convergent
    convergent_computer = ConvergentCausality()
    incoming_graph_conv = convergent_computer.compute(mock_source, mock_neighbors, mock_num_points)
    expected_conv = [[], [0], [1, 3], [0]] # Flow is 0->1, 0->3, 1->2, 3->2
    assert incoming_graph_conv == expected_conv, f"Convergent test failed! Got {incoming_graph_conv}"
    cprint("Convergent Test Passed!", 'green')

    # Test 2: Divergent
    divergent_computer = DivergentCausality()
    incoming_graph_div = divergent_computer.compute(mock_source, mock_neighbors, mock_num_points)
    expected_div = [[1, 3], [2], [], [2]] # Flow is 1->0, 3->0, 2->1, 2->3
    assert incoming_graph_div == expected_div, f"Divergent test failed! Got {incoming_graph_div}"
    cprint("Divergent Test Passed!", 'green')

    cprint("\n--- All tests for causality.py passed! ---", 'yellow', attrs=['bold'])
