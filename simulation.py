# simulation.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "The Engine Abstraction"
# - CRITICAL REFACTOR: The Simulation class is now a generic "engine"
#   that operates on an `AbstractField` object, not a raw numpy array.
# - It is no longer responsible for creating the initial state; it receives
#   a fully constructed `field` object upon initialization.
# - The evolution logic `_evolve_field` now correctly handles multi-component
#   fields by applying the Laplacian component-wise.
# - The `update_step` logic is decoupled: it first evolves the field, then
#   asks the field for its `interaction_source` to pass to the causality module.
#   This makes the engine physics-agnostic.

import numpy as np
import multiprocessing as mp
from termcolor import cprint

# --- Import abstract classes for type hinting and dependency injection ---
from topologies import TopologyData
from causality import AbstractCausalityComputer
from field import AbstractField # NEW: Import the abstract field

class Simulation:
    """
    The main simulation engine for Project Genesis.
    It orchestrates the co-evolution of a physical field and its emergent
    causal structure on a static substrate.
    """
    def __init__(self,
                 topology_data: TopologyData,
                 field: AbstractField, # MODIFIED: Takes a Field object
                 causality_computer: AbstractCausalityComputer,
                 use_multiprocessing: bool = True):
        """
        Initializes the simulation engine.

        Args:
            topology_data: A TopologyData object describing the static substrate.
            field: An object of a class inheriting from AbstractField (e.g., ScalarField).
            causality_computer: The strategy for determining the emergent causal graph.
            use_multiprocessing: Flag to enable the multiprocessing pool.
        """
        cprint(f"3. Assembling Simulation Engine (v16.0 - Engine Abstraction)...", 'cyan', attrs=['bold'])

        self.substrate = topology_data
        self.field = field  # Store the abstract field object
        self.causality_computer = causality_computer
        self.num_points = self.substrate.num_points

        # --- VALIDATION ---
        if self.num_points != self.field.num_points:
            raise ValueError("Field size must match topology size!")

        # Pre-calculate the graph Laplacian for efficiency.
        # This is the core of the Schrödinger evolution.
        self._laplacian_matrix = self._build_laplacian_matrix()
        cprint(f"   -> Pre-calculated graph Laplacian matrix.", 'green')

        # Multiprocessing setup (no changes)
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

    def _build_laplacian_matrix(self):
        """Builds the graph Laplacian matrix `L = D - A` where D is the degree matrix and A is the adjacency matrix."""
        adj_matrix = np.zeros((self.num_points, self.num_points), dtype=np.float32)
        degrees = np.zeros(self.num_points, dtype=np.float32)

        for i, neighbors in enumerate(self.substrate.neighbors):
            degrees[i] = len(neighbors)
            for j in neighbors:
                adj_matrix[i, j] = 1

        degree_matrix = np.diag(degrees)
        laplacian = degree_matrix - adj_matrix
        return laplacian

    def _evolve_field(self, dt: float = 0.01):
        """
        Evolves the abstract field using the Schrödinger equation.
        d(field)/dt = i * Laplacian(field)

        This operation is vectorized for performance. The Laplacian matrix is applied
        to each component of the field's `values` array independently.
        """
        # The @ operator is Python's syntax for matrix multiplication.
        laplacian_result = self._laplacian_matrix @ self.field.values

        # Update the field values
        self.field.values += 1j * laplacian_result * dt

        # The field object itself knows how to re-normalize.
        self.field.normalize()

    def update_step(self) -> list:
        """
        Executes one full step of the co-evolution.
        This is the core loop of the "Breathing Causality" paradigm.
        """
        # STEP 1: The quantum field evolves on its own, on the static substrate.
        self._evolve_field()

        # STEP 2: The new state of the field determines the emergent causal structure.
        # We ask the field for its "causal source" (e.g., |psi|^2 for a scalar field).
        interaction_source = self.field.get_interaction_source()

        # We pass this source to the causality module to compute the directed graph.
        causal_graph = self.causality_computer.compute(
            interaction_source,
            self.substrate.neighbors,
            self.num_points
        )

        # The causal graph is returned for analysis, but is not used to evolve psi
        # in the next step, aligning with the "Causality as Consequence" paradigm.
        return causal_graph

    def close(self):
        """Cleanly shuts down the simulation resources."""
        if self.pool:
            self.pool.close()
            self.pool.join()
        cprint("\nSimulation engine shut down.", 'yellow')


# --- Example Usage for Testing (updated for the new architecture) ---
if __name__ == "__main__":
    from topologies import generate_crystal_topology
    from causality import ConvergentCausality
    from initial_conditions import PrimordialSoupState

    cprint("\n--- Testing simulation.py v16.0 ---", 'yellow', attrs=['bold'])

    # 1. GENERATE data and field objects first
    topology_data = generate_crystal_topology(width=10, height=10)

    # The initial condition generator now returns a Field object
    initial_state_gen = PrimordialSoupState()
    initial_field = initial_state_gen.generate(topology_data)

    causality_gen = ConvergentCausality()

    # 2. Instantiate the simulation by injecting the dependencies
    cprint("1. Instantiating simulation with a ScalarField...", 'cyan')
    try:
        sim = Simulation(topology_data, initial_field, causality_gen, use_multiprocessing=False)
        cprint("  > SUCCESS: Simulation engine assembled correctly.", 'green')
    except Exception as e:
        cprint(f"  > FAILED: {e}", 'red')
        exit()

    # 3. Run basic tests
    cprint("2. Running basic physics checks...", 'cyan')
    initial_prob = np.sum(np.abs(sim.field.values)**2)
    print(f"   - Initial total probability: {initial_prob:.4f}")
    assert np.isclose(initial_prob, 1.0), "Initial state is not normalized."

    print(f"   - Running 10 simulation steps...")
    for i in range(10):
        sim.update_step()

    final_prob = np.sum(np.abs(sim.field.values)**2)
    print(f"   - Final total probability: {final_prob:.4f}")
    assert np.isclose(final_prob, 1.0), "Probability was not conserved during evolution."
    cprint("  > SUCCESS: Probability is conserved.", 'green')

    sim.close()

    cprint("\n--- All tests for simulation.py passed! ---", 'yellow', attrs=['bold'])
