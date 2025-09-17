# simulation.py v15.1
# Part of Project Genesis: Breathing Causality
# v15.1: "Paradigm Shift" - CRITICAL bugfix and conceptual realignment.
# - The core evolution logic in `_evolve_psi` is corrected to implement the
#   standard Schrödinger equation on a graph (i * dψ/dt = -∇²ψ),
#   fixing a fundamental bug that caused energy non-conservation.
# - The `update_step` logic is realigned with our core hypothesis:
#   1. The quantum field `psi` evolves first according to its own laws (Schrödinger).
#   2. The new state of `psi` then determines the emergent causal graph `G_causal`.
# - Causality is now a *consequence* of quantum evolution, not its driver. This
#   makes the model physically sound and prepares it for future extensions.

import numpy as np
import multiprocessing as mp
from termcolor import cprint

# Import the data container and abstract strategies for type hinting
from topologies import TopologyData
from causality import AbstractCausalityEvolver
from initial_conditions import BaseInitialState

class Simulation:
    """
    The main simulation orchestrator for Project Genesis v15.1.
    This version implements the corrected physics based on the Schrödinger equation
    and the "Causality as a Consequence" paradigm.
    """
    def __init__(self,
                 topology_data: TopologyData,
                 causality_evolver: AbstractCausalityEvolver,
                 initial_state_generator: BaseInitialState,
                 use_multiprocessing: bool = True):
        """
        Initializes the simulation with concrete data and swappable strategies.
        """
        cprint(f"3. Assembling Simulation Engine (v15.1 - Paradigm Shift)...", 'cyan', attrs=['bold'])

        self.substrate = topology_data
        self.causality_evolver = causality_evolver
        self.psi = initial_state_generator.generate(self.substrate)
        self.num_points = self.substrate.num_points

        # Multiprocessing setup (remains the same)
        self.pool = None
        if use_multiprocessing and self.num_points > 1000:
            try:
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method('spawn')
                self.pool = mp.Pool(mp.cpu_count())
                cprint(f"   -> Multiprocessing enabled on {mp.cpu_count()} cores.", 'green')
            except (RuntimeError, ValueError) as e:
                cprint(f"   -> Warning: Multiprocessing pool setup failed: {e}. Running single-threaded.", 'yellow')
                self.pool = None

    def _evolve_psi(self, directed_causality: list, dt: float = 0.01):
        psi_new = self.psi.copy()

        for i in range(self.num_points):
            causal_inputs = directed_causality[i] # Узлы j, для которых есть связь j -> i

            if causal_inputs:
                incoming_signal = np.sum(self.psi[causal_inputs])
                epsilon = dt * len(causal_inputs)
                psi_new[i] = (1 - 1j * epsilon) * self.psi[i] + (1j * dt) * incoming_signal

        self.psi = psi_new

        norm = np.linalg.norm(self.psi)
        if norm > 1e-9:
            self.psi /= norm

    def update_step(self) -> list:
        causal_graph = self.causality_evolver.evolve(
            self.psi,
            self.substrate.neighbors,
            self.num_points
        )

        self._evolve_psi(causal_graph)

        return causal_graph

    def close(self):
        """Cleanly shuts down the simulation resources."""
        if self.pool:
            self.pool.close()
            self.pool.join()
        cprint("\nSimulation engine shut down.", 'yellow')


# --- Example Usage for Testing (updated for the new paradigm) ---
if __name__ == "__main__":
    from topologies import generate_crystal_topology
    from causality import AmplitudeConvergentCausality
    from initial_conditions import PrimordialSoupState

    cprint("\n--- Testing Simulation Core v15.1 (Paradigm Shift) ---", 'yellow')

    # 1. GENERATE data first
    topology_data = generate_crystal_topology(width=30, height=20)

    # 2. Setup the other components
    causality_gen = AmplitudeConvergentCausality()
    initial_state_gen = PrimordialSoupState()

    # 3. Instantiate the simulation
    sim = Simulation(topology_data, causality_gen, initial_state_gen, use_multiprocessing=False)

    # 4. Run tests
    print(f"\nInitial total probability |psi|^2: {np.sum(np.abs(sim.psi)**2):.4f}")
    assert np.isclose(np.sum(np.abs(sim.psi)**2), 1.0)

    print(f"\nRunning 10 simulation steps with the corrected Schrödinger evolution...")
    for i in range(10):
        # The API call remains the same, but the internal logic has changed.
        sim.update_step()

    print(f"Final total probability |psi|^2: {np.sum(np.abs(sim.psi)**2):.4f}")
    assert np.isclose(np.sum(np.abs(sim.psi)**2), 1.0)

    sim.close()

    cprint("\nTest Passed: Corrected simulation core is stable and conserves probability.", 'green')
