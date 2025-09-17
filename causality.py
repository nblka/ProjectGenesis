# causality.py v13.1
# Part of Project Genesis: Breathing Causality
# v13.1: "Conceptual Realignment" - The logic of this module is unchanged,
#        but its role in the simulation has been clarified.
# - This module no longer DRIVES the evolution of psi.
# - Instead, it ANALYZES the state of psi at each frame to determine the
#   EMERGENT causal graph G_causal.
# - It contains competing hypotheses for how the quantum field might
#   induce a directed causal structure on the static substrate.

import numpy as np
from abc import ABC, abstractmethod
from termcolor import cprint

class AbstractCausalityEvolver(ABC):
    """
    Abstract Base Class for all emergent causality strategies.

    Defines the interface for modules that compute the instantaneous
    directed causal graph G_causal(t) from the quantum field state psi(t)
    and the static undirected substrate. The result is an adjacency list
    representing the "flow" of causality for that moment.
    """
    def __init__(self):
        self.strategy_name = "Abstract"

    @abstractmethod
    def evolve(self, psi: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        """
        Computes the INCOMING directed adjacency list for the current frame.

        Args:
            psi (np.ndarray): The current state of the quantum field.
            undirected_neighbors (list): The static substrate's undirected adjacency list.
            num_points (int): The total number of points in the substrate.

        Returns:
            list: An INCOMING directed adjacency list, where incoming_neighbors[i]
                  contains a list of nodes `j` such that the emergent causal
                  edge is defined as j -> i.
        """
        pass

class AmplitudeConvergentCausality(AbstractCausalityEvolver):
    """
    HYPOTHESIS 1: "Convergent Flow" or "Uphill Causality"

    Causality flows from lower amplitude to higher amplitude regions. This
    can be interpreted as information concentrating towards areas of high
    "presence" or mass-energy, akin to a gravitational pull. In this model,
    stable particles would be "causal sinks".
    """
    def __init__(self):
        super().__init__()
        self.strategy_name = "Convergent (Uphill)"
        cprint(f"   -> Causality Strategy: {self.strategy_name}", 'cyan')

    def evolve(self, psi: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        """

        Generates an incoming directed graph based on the rule:
        An edge j -> i exists if the amplitude at i is greater than at j.
        """
        incoming_neighbors = [[] for _ in range(num_points)]
        amplitudes_sq = np.abs(psi)**2

        for i in range(num_points):
            amp_i = amplitudes_sq[i]
            # Iterate through the neighbors of node i
            for j in undirected_neighbors[i]:
                # To avoid double-counting and self-loops, we only process pairs where i > j
                if i > j:
                    amp_j = amplitudes_sq[j]

                    if amp_i > amp_j:
                        # Amplitude at i is higher, so flow is j -> i.
                        # We add j to the list of incoming neighbors for i.
                        incoming_neighbors[i].append(j)
                    elif amp_j > amp_i:
                        # Amplitude at j is higher, so flow is i -> j.
                        # We add i to the list of incoming neighbors for j.
                        incoming_neighbors[j].append(i)
                    # If amplitudes are equal, no causal edge is formed.

        return incoming_neighbors

class AmplitudeDivergentCausality(AbstractCausalityEvolver):
    """
    HYPOTHESIS 2: "Divergent Flow" or "Downhill Causality"

    Causality flows from higher amplitude to lower amplitude regions. This
    can be interpreted as presence/energy dissipating or flowing outwards,
    like heat or a radiating source. In this model, stable particles would
    be "causal sources".
    """
    def __init__(self):
        super().__init__()
        self.strategy_name = "Divergent (Downhill)"
        cprint(f"   -> Causality Strategy: {self.strategy_name}", 'cyan')

    def evolve(self, psi: np.ndarray, undirected_neighbors: list, num_points: int) -> list:
        incoming_neighbors = [[] for _ in range(num_points)]
        amplitudes_sq = np.abs(psi)**2

        for i in range(num_points):
            amp_i = amplitudes_sq[i]
            for j in undirected_neighbors[i]:
                if i > j:
                    amp_j = amplitudes_sq[j]

                    if amp_i > amp_j:
                        incoming_neighbors[j].append(i)
                    elif amp_j > amp_i:
                        incoming_neighbors[i].append(j)

        return incoming_neighbors

# --- Example Usage for Testing ---
# This block is for verifying the logic of the module independently.
if __name__ == "__main__":
    cprint("\n--- Testing CausalityEvolver v13.1 (Conceptual Realignment) ---", 'yellow')

    # Test Setup
    # --- FIX: Correctly define a square graph (0-1, 1-2, 2-3, 3-0) ---
    mock_neighbors = [[1, 3], [0, 2], [1, 3], [0, 2]]
    mock_num_points = 4
    # Amplitudes: 0: 1.0, 1: 4.0, 2: 16.0, 3: 9.0
    mock_psi = np.array([1.0, 2.0, 4.0, 3.0], dtype=complex)
    cprint(f"Test substrate: square graph. Amplitudes: {np.abs(mock_psi)**2}", 'white')

    # Test 1: Convergent (Flows "uphill" towards higher amplitude)
    convergent_evolver = AmplitudeConvergentCausality()
    incoming_graph_conv = convergent_evolver.evolve(mock_psi, mock_neighbors, mock_num_points)
    # --- FIX: Recalculate expected result for the correct graph ---
    # Expected flow: 0->1, 0->3, 1->2, 3->2
    expected_conv = [[], [0], [1, 3], [0]]
    assert incoming_graph_conv == expected_conv, f"Convergent test failed! Got {incoming_graph_conv}, expected {expected_conv}"
    cprint("Convergent Test Passed!", 'green')

    # Test 2: Divergent (Flows "downhill" away from higher amplitude)
    divergent_evolver = AmplitudeDivergentCausality()
    incoming_graph_div = divergent_evolver.evolve(mock_psi, mock_neighbors, mock_num_points)
    # --- FIX: Recalculate expected result for the correct graph ---
    # Expected flow: 1->0, 3->0, 2->1, 2->3
    expected_div = [[1, 3], [2], [], [2]]
    assert incoming_graph_div == expected_div, f"Divergent test failed! Got {incoming_graph_div}, expected {expected_div}"
    cprint("Divergent Test Passed!", 'green')
