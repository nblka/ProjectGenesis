# simulation.py
import numpy as np
import multiprocessing as mp
from topologies import CrystalTopology
from initial_conditions import BaseInitialState

def calculate_laplacian_chunk(indices_chunk, psi, neighbors):
    laplacian_chunk = np.zeros(len(indices_chunk), dtype=complex)
    for i, real_idx in enumerate(indices_chunk):
        node_neighbors = neighbors[real_idx]
        if node_neighbors:
            laplacian_chunk[i] = np.sum(psi[node_neighbors] - psi[real_idx])
    return laplacian_chunk

class Simulation:
    def __init__(self, topology: CrystalTopology, initial_state: BaseInitialState, use_multiprocessing=True):
        print(f"2. Composing Universe Engine...")
        self.topology = topology
        self.psi = initial_state.generate(self.topology)

        # Для рендеринга мы можем вычислить симплексы один раз
        self.simplices = self.topology.get_simplices()

        self.pool = None
        if use_multiprocessing:
            self.pool = mp.Pool(mp.cpu_count())
            self.point_chunks = np.array_split(np.arange(self.topology.num_points), mp.cpu_count())
            print(f"   -> Multiprocessing enabled on {mp.cpu_count()} cores.")

    def _evolve_psi(self, dt=0.01):
        if self.pool:
            args = [(chunk, self.psi, self.topology.neighbors) for chunk in self.point_chunks]
            results = self.pool.starmap(calculate_laplacian_chunk, args)
            laplacian = np.concatenate(results)
        else:
            laplacian = calculate_laplacian_chunk(np.arange(self.topology.num_points), self.psi, self.topology.neighbors)

        self.psi += 1j * laplacian * dt
        norm = np.linalg.norm(self.psi)
        if norm > 1e-9:
            self.psi /= norm

    def update_step(self):
        """Один шаг симуляции. Теперь здесь только эволюция поля."""
        self._evolve_psi()
        # TODO: В будущем здесь будет опциональный вызов _try_topology_flips

    def close(self):
        if self.pool: self.pool.close(); self.pool.join()
