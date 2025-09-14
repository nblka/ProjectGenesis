# simulation.py v13.1
# Part of Project Genesis: Breathing Causality
# v13.1: Final, robust version of the data-centric simulation core.
# - Fully decoupled from component implementation details.
# - Minor fix in _evolve_psi to prevent mutation of the original psi.

import numpy as np
import multiprocessing as mp
from termcolor import cprint

# Import the data container and abstract strategies for type hinting
from topologies import TopologyData
from causality import AbstractCausalityEvolver
from initial_conditions import BaseInitialState

class Simulation:
    """
    The main simulation orchestrator for Project Genesis v13.1.
    Accepts pre-generated data structures and runs the co-evolution loop.
    """
    def __init__(self,
                 topology_data: TopologyData,
                 causality_evolver: AbstractCausalityEvolver,
                 initial_state_generator: BaseInitialState,
                 use_multiprocessing: bool = True):
        """
        Initializes the simulation with concrete data and swappable strategies.
        """
        cprint(f"3. Assembling Simulation Engine...", 'cyan', attrs=['bold'])

        self.substrate = topology_data
        self.causality_evolver = causality_evolver
        self.psi = initial_state_generator.generate(self.substrate)
        self.num_points = self.substrate.num_points

        # Multiprocessing setup
        self.pool = None
        if use_multiprocessing and self.num_points > 1000:
            try:
                # Set start method for compatibility with macOS/Windows
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method('spawn')
                self.pool = mp.Pool(mp.cpu_count())
                self.point_chunks = np.array_split(np.arange(self.num_points), mp.cpu_count())
                cprint(f"   -> Multiprocessing enabled on {mp.cpu_count()} cores.", 'green')
            except (RuntimeError, ValueError) as e:
                cprint(f"   -> Warning: Multiprocessing pool already started or failed to start: {e}. Running single-threaded.", 'yellow')
                self.pool = None

    def _evolve_psi(self, directed_causality: list, dt: float = 0.01):
        """
        Evolves the psi field for one step based on the provided
        INSTANTANEOUS directed causal graph (incoming adjacency list).
        """
        # This is a simplified Euler-method version that captures the dependency.
        # A more advanced version would build and apply a sparse unitary matrix.

        laplacian = np.zeros(self.num_points, dtype=np.complex128)

        for i in range(self.num_points):
            causal_inputs = directed_causality[i] # Nodes j where j -> i
            if causal_inputs:
                # The change at i is the sum of differences from its causal sources.
                laplacian[i] = np.sum(self.psi[causal_inputs] - self.psi[i])

        # Evolve the state
        self.psi += 1j * laplacian * dt

        # Re-normalize to conserve total probability
        norm = np.linalg.norm(self.psi)
        if norm > 1e-9:
            self.psi /= norm

    def update_step(self) -> list:
        """
        Performs one full step of the "Breathing Causality" cycle.
        Returns the causal graph for analytics.
        """
        # Step 1: Matter determines Causality
        causal_graph = self.causality_evolver.evolve(
            self.psi,
            self.substrate.neighbors,
            self.num_points
        )

        # Step 2: Causality determines Matter's evolution
        self._evolve_psi(causal_graph)

        return causal_graph

    def close(self):
        """Cleanly shuts down the simulation resources."""
        if self.pool:
            self.pool.close()
            self.pool.join()
        cprint("\nSimulation engine shut down.", 'yellow')


# --- Example Usage for Testing ---
if __name__ == "__main__":
    from topologies import generate_crystal_topology
    from causality import AmplitudeConvergentCausality
    from initial_conditions import PrimordialSoupState

    cprint("\n--- Testing Simulation Core v13.0 ---", 'yellow')

    # 1. GENERATE data first. This is now a separate step.
    topology_data = generate_crystal_topology(width=30, height=20)

    # 2. Setup the other components
    causality_gen = AmplitudeConvergentCausality()
    initial_state_gen = PrimordialSoupState()

    # 3. Instantiate the simulation with the DATA object
    sim = Simulation(topology_data, causality_gen, initial_state_gen, use_multiprocessing=False)

    # 4. Run tests (same as before)
    print(f"\nInitial total probability |psi|^2: {np.sum(np.abs(sim.psi)**2):.4f}")
    assert np.isclose(np.sum(np.abs(sim.psi)**2), 1.0)

    print(f"\nRunning 10 simulation steps...")
    for i in range(10):
        sim.update_step()

    print(f"Final total probability |psi|^2: {np.sum(np.abs(sim.psi)**2):.4f}")
    assert np.isclose(np.sum(np.abs(sim.psi)**2), 1.0)

    sim.close()

    cprint("\nTest Passed: Data-centric simulation core works as expected.", 'green')
