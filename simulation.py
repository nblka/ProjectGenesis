# simulation.py v19.0
# Part of Project Genesis: Breathing Causality
# v19.0: "Local Catalysis & Spontaneous Symmetry Breaking"
# - FUNDAMENTAL SHIFT 2.0: The global, entropy-based self-regulation is replaced
#   by a LOCAL, self-catalyzing mechanism.
# - The `interaction_strength` is now a FIELD, `strength_field[i]`, calculated for each node.
# - The strength at a node is now proportional to a measure of LOCAL ORDER around it.
#   This allows small, random fluctuations of order to amplify themselves,
#   providing a mechanism for spontaneous symmetry breaking and particle formation.
# - This model is far more physically realistic, resembling mechanisms of structure
#   formation in condensed matter physics.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import multiprocessing as mp
from termcolor import cprint

# --- Import abstract classes ---
from topologies import TopologyData
from causality import AbstractCausalityComputer
from field import AbstractField

class Simulation:
    """
    The main simulation engine for Project Genesis.
    v19.0 implements a local, self-catalyzing interaction mechanism.
    """
    def __init__(self,
                 topology_data: TopologyData,
                 field: AbstractField,
                 causality_computer: AbstractCausalityComputer,
                 base_interaction_strength: float = 200.0, # This can be higher now
                 use_multiprocessing: bool = True):
        
        cprint(f"3. Assembling Simulation Engine (v19.0 - Local Catalysis)...", 'cyan', attrs=['bold'])

        self.substrate = topology_data
        self.field = field
        self.causality_computer = causality_computer
        self.num_points = self.substrate.num_points
        self.base_interaction_strength = base_interaction_strength

        if self.num_points != self.field.num_points:
            raise ValueError("Field size must match topology size!")

        self._laplacian_matrix_sparse = sp.csr_matrix(self._build_laplacian_matrix())
        cprint(f"   -> Pre-calculated sparse graph Laplacian (Kinetic Term).", 'green')
        
        # Pre-build a list of neighbor indices for efficient local calculations
        self._neighbor_indices_list = [np.array(n, dtype=int) for n in self.substrate.neighbors]
        
        # (Multiprocessing setup is unchanged)
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
        self.pool = None

    def _build_laplacian_matrix(self):
        # (This method is unchanged)
        adj_matrix = np.zeros((self.num_points, self.num_points), dtype=np.float32)
        degrees = np.zeros(self.num_points, dtype=np.float32)
        for i, neighbors in enumerate(self.substrate.neighbors):
            degrees[i] = len(neighbors)
            for j in neighbors: adj_matrix[i, j] = 1
        degree_matrix = np.diag(degrees)
        return degree_matrix - adj_matrix

    def _calculate_local_order_field(self, interaction_source: np.ndarray) -> np.ndarray:
        """
        Calculates a measure of local order for each node.
        NEW in v19.0.
        
        Here, we define 'order' as the local concentration. For each node `i`, we calculate
        the coefficient of variation of the source field among itself and its neighbors.
        A high CV means a 'peaky', ordered region. A low CV means a 'flat', chaotic region.
        """
        local_order = np.zeros(self.num_points, dtype=np.float32)
        
        # This is a good candidate for parallelization if it's slow
        for i in range(self.num_points):
            neighbor_indices = self._neighbor_indices_list[i]
            # Include the node itself in its local neighborhood
            local_indices = np.append(neighbor_indices, i)
            
            local_source_values = interaction_source[local_indices]
            
            mean_val = np.mean(local_source_values)
            if mean_val > 1e-12:
                std_dev = np.std(local_source_values)
                local_order[i] = std_dev / mean_val
            # If mean is zero, order is zero.
        
        return local_order

    def _evolve_field_locally_unitary(self, dt: float = 0.01):
        """
        Evolves the field using a Crank-Nicolson method with a LOCAL and DYNAMIC Hamiltonian.
        """
        interaction_source = self.field.get_interaction_source()
        
        # Step 1: Calculate the local order field.
        local_order_field = self._calculate_local_order_field(interaction_source)
        
        # Step 2: Calculate the dynamic, local interaction strength field.
        # This is our catalysis function. We use a simple power law (e.g., squared)
        # to make the feedback strong. Ordered regions get much stronger interaction.
        dynamic_strength_field = self.base_interaction_strength * (local_order_field ** 2)
        
        # Step 3: Construct the Hamiltonian using this dynamic strength field.
        H_kinetic = self._laplacian_matrix_sparse
        H_potential = sp.diags(-dynamic_strength_field * interaction_source)
        H = H_kinetic + H_potential
        
        # Step 4: Solve the system using Crank-Nicolson (as in v17.1).
        I = sp.identity(self.num_points, dtype=np.complex128)
        A = I + 0.5j * H * dt
        B = I - 0.5j * H * dt
        
        for c in range(self.field.num_components):
            psi_current = self.field.values[:, c]
            b = B @ psi_current
            psi_next = spsolve(A, b, use_umfpack=True) # UMFPACK is often faster for such systems
            self.field.values[:, c] = psi_next

    def update_step(self) -> list:
        """Executes one full step of the simulation."""
        self._evolve_field_locally_unitary()
        interaction_source = self.field.get_interaction_source()
        causal_graph = self.causality_computer.compute(
            interaction_source,
            self.substrate.neighbors,
            self.num_points
        )
        return causal_graph

    def close(self):
        """Cleanly shuts down the simulation resources."""
        if self.pool:
            self.pool.close()
            self.pool.join()
        cprint("\nSimulation engine shut down.", 'yellow')


# --- Testing Block ---
if __name__ == "__main__":
    from topologies import generate_crystal_topology
    from causality import ConvergentCausality
    from initial_conditions import PrimordialSoupState
    import matplotlib.pyplot as plt

    cprint("\n--- Testing simulation.py v18.0 (Self-Regulation) ---", 'yellow', attrs=['bold'])

    # We can now use a single, reasonable base_strength and expect interesting results
    BASE_STRENGTH = 100.0

    topo = generate_crystal_topology(width=20, height=20)
    field = PrimordialSoupState().generate(topo)
    causality = ConvergentCausality()

    sim = Simulation(
        topo, 
        field, 
        causality, 
        base_interaction_strength=BASE_STRENGTH, 
        use_multiprocessing=False
    )

    initial_prob = np.sum(np.abs(sim.field.values)**2)
    print(f"Initial total probability: {initial_prob:.12f}")
    assert np.isclose(initial_prob, 1.0)

    # We can also track how the dynamic strength changes
    strength_history = []
    
    print(f"\nRunning 100 steps with dynamic interaction strength (base={BASE_STRENGTH})...")
    for i in range(100):
        # We can extract the dynamic strength for logging if we modify the function slightly
        # For now, just run the simulation
        sim.update_step()
    
    final_prob = np.sum(np.abs(sim.field.values)**2)
    print(f"Final total probability after 100 steps: {final_prob:.12f}")

    assert np.allclose(final_prob, 1.0, atol=1e-9), "Probability conservation failed!"
    cprint("  > SUCCESS: Probability is conserved with the new dynamic Hamiltonian.", 'green')
    
    sim.close()

    cprint("\n--- Test Passed! The self-regulating engine is stable. ---", 'yellow', attrs=['bold'])