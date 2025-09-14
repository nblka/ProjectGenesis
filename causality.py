# causality.py v13.0
# Part of Project Genesis: Breathing Causality
# v13.0: Final, robust version for the data-centric architecture.
# - Contains the abstract base class for causality evolution.
# - Provides two concrete, competing strategies: Convergent and Divergent flow.
# - API is stabilized to accept psi and substrate components directly.

import numpy as np
from abc import ABC, abstractmethod
from termcolor import cprint

class AbstractCausalityEvolver(ABC):
    """
    Abstract Base Class for all causality evolution strategies.
    Defines the interface for modules that determine the instantaneous
    directed causal graph G(n) from the quantum field state psi(n)
    and the static undirected substrate.
    """
    def __init__(self):
        self.strategy_name = "Abstract"

    @abstractmethod
    def evolve(self, psi: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        """
        The core method that computes the INCOMING directed adjacency list.

        Args:
            psi (np.ndarray): The current state of the quantum field.
            undirected_neighbors (list): The static substrate's undirected adjacency list.
            num_points (int): The total number of points in the substrate.

        Returns:
            list: An INCOMING directed adjacency list, where incoming_neighbors[i]
                  contains a list of nodes `j` such that the causal edge is j -> i.
        """
        pass

class AmplitudeConvergentCausality(AbstractCausalityEvolver):
    """
    HYPOTHESIS 1: "Convergent Flow" or "Uphill Causality"
    Causality flows from lower amplitude to higher amplitude regions.
    Represents information concentrating towards areas of high "presence".
    """
    def __init__(self):
        super().__init__()
        self.strategy_name = "Convergent (Uphill)"
        cprint(f"   -> Causality Strategy: {self.strategy_name}", 'cyan')


    def evolve(self, psi: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        """
        Generates an incoming directed graph based on the rule:
        An edge j -> i exists if amp_i > amp_j.
        """
        incoming_neighbors = [[] for _ in range(num_points)]
        amplitudes_sq = np.abs(psi)**2

        for i in range(num_points):
            amp_i = amplitudes_sq[i]
            for j in undirected_neighbors[i]:
                if i > j: continue

                amp_j = amplitudes_sq[j]

                if amp_j > amp_i:
                    incoming_neighbors[j].append(i) # Edge is i -> j
                elif amp_i > amp_j:
                    incoming_neighbors[i].append(j) # Edge is j -> i

        return incoming_neighbors

class AmplitudeDivergentCausality(AbstractCausalityEvolver):
    """
    HYPOTHESIS 2: "Divergent Flow" or "Downhill Causality"
    Causality flows from higher amplitude to lower amplitude regions.
    Represents energy/presence dissipating or flowing outwards, like heat.
    """
    def __init__(self):
        super().__init__()
        self.strategy_name = "Divergent (Downhill)"
        cprint(f"   -> Causality Strategy: {self.strategy_name}", 'cyan')

    def evolve(self, psi: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        """
        Generates an incoming directed graph based on the rule:
        An edge j -> i exists if amp_j > amp_i.
        """
        incoming_neighbors = [[] for _ in range(num_points)]
        amplitudes_sq = np.abs(psi)**2

        for i in range(num_points):
            amp_i = amplitudes_sq[i]
            for j in undirected_neighbors[i]:
                if i > j: continue

                amp_j = amplitudes_sq[j]

                if amp_j < amp_i:
                    incoming_neighbors[j].append(i) # Edge is i -> j
                elif amp_i < amp_j:
                    incoming_neighbors[i].append(j) # Edge is j -> i

        return incoming_neighbors


# --- Example Usage for Testing ---
if __name__ == "__main__":
    # The test block also needs to be updated to use the new API
    cprint("\n--- Testing CausalityEvolver v12.3 (API Fix) ---", 'yellow')

    # Test Setup
    mock_neighbors = [[1], [0, 2], [1, 3], [2]]
    mock_num_points = 4
    mock_psi = np.array([1.0, 2.0, 4.0, 3.0], dtype=complex)

    # Test 1: Convergent
    convergent_evolver = AmplitudeConvergentCausality()
    incoming_graph_conv = convergent_evolver.evolve(mock_psi, mock_neighbors, mock_num_points)
    expected_conv = [[], [0], [1, 3], []]
    assert incoming_graph_conv == expected_conv, "Convergent test failed!"
    cprint("Convergent Test Passed!", 'green')

    # Test 2: Divergent
    divergent_evolver = AmplitudeDivergentCausality()
    incoming_graph_div = divergent_evolver.evolve(mock_psi, mock_neighbors, mock_num_points)
    expected_div = [[1], [2], [], [2]] # Logic re-verified: 1->0, 2->1, 2->3
    assert incoming_graph_div == expected_div, "Divergent test failed!"
    cprint("Divergent Test Passed!", 'green')
