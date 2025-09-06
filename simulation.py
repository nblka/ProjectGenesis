# simulation.py v6.2
"""
Simulation Core Module for Project Genesis
-------------------------------------------
- CRITICAL FIX: Restored the missing core evolution methods (_evolve_psi, etc.)
  that were lost during previous refactoring.
- FEATURE: Includes reflecting boundary conditions to keep the simulation contained.
- FEATURE: Implements velocity and damping for more realistic node movement.
"""
import numpy as np
from scipy.spatial import Delaunay
import multiprocessing as mp

def calculate_laplacian_chunk(indices, psi, vertex_neighbors):
    laplacian_chunk = np.zeros(len(indices), dtype=complex)
    for i, real_idx in enumerate(indices):
        neighbors = vertex_neighbors[real_idx]
        if neighbors: laplacian_chunk[i] = np.sum(psi[neighbors] - psi[real_idx])
    return laplacian_chunk

class Simulation:
    def __init__(self, size=40, num_points=500, use_multiprocessing=True):
        print(f"1. Creating initial complex with {num_points} nodes...")
        self.size = size
        self.num_points = num_points
        self.points = np.random.rand(self.num_points, 2) * (self.size - 2) + 1
        self.velocities = np.random.randn(self.num_points, 2) * 0.01
        self.tri, self.simplices, self.vertex_neighbors = None, None, None
        self._update_triangulation()
        self.psi = np.zeros(self.num_points, dtype=complex)
        self._initialize_psi()
        self.pool = None
        if use_multiprocessing and num_points > 2000:
            self.pool = mp.Pool(mp.cpu_count())
            print(f"   -> Multiprocessing enabled on {mp.cpu_count()} cores.")
            self.point_chunks = np.array_split(np.arange(self.num_points), mp.cpu_count())
        else:
            print("   -> Running in single-threaded mode.")

    def _initialize_psi(self):
        raise NotImplementedError("Subclasses must implement _initialize_psi!")

    def _update_triangulation(self):
        self.tri = Delaunay(self.points, qhull_options="Qbb Qc Qz")
        self.simplices = self.tri.simplices
        self.vertex_neighbors = [[] for _ in range(self.num_points)]
        for simplex in self.simplices:
            for i in range(3):
                p1, p2, p3 = simplex[i], simplex[(i + 1) % 3], simplex[(i + 2) % 3]
                self.vertex_neighbors[p1].extend([p2, p3])
        for i in range(self.num_points): self.vertex_neighbors[i] = list(set(self.vertex_neighbors[i]))

    def _evolve_psi(self, dt=0.01):
        if self.pool:
            args = [(chunk, self.psi, self.vertex_neighbors) for chunk in self.point_chunks]
            results = self.pool.starmap(calculate_laplacian_chunk, args)
            laplacian = np.concatenate(results)
        else:
            laplacian = calculate_laplacian_chunk(np.arange(self.num_points), self.psi, self.vertex_neighbors)
        self.psi += 1j * laplacian * dt
        self.psi /= np.linalg.norm(self.psi)

    def _apply_boundary_conditions(self):
        for i in range(2):
            out_of_bounds_low = self.points[:, i] < 0
            out_of_bounds_high = self.points[:, i] > self.size
            self.points[out_of_bounds_low, i] *= -1
            self.velocities[out_of_bounds_low, i] *= -1
            self.points[out_of_bounds_high, i] = 2 * self.size - self.points[out_of_bounds_high, i]
            self.velocities[out_of_bounds_high, i] *= -1

    def _try_geometry_flips(self, num_flips=10, strength=0.003):
        for _ in range(num_flips):
            if len(self.simplices) == 0: continue
            t1_idx = np.random.randint(len(self.simplices))
            edge_to_flip = np.random.randint(3)
            n_idx = self.tri.neighbors[t1_idx][edge_to_flip]
            if n_idx == -1: continue
            p1, p2 = [v for i, v in enumerate(self.simplices[t1_idx]) if i != edge_to_flip]
            p3_candidates = [v for v in self.simplices[t1_idx] if v not in [p1, p2]]
            p4_candidates = [v for v in self.simplices[n_idx] if v not in [p1, p2]]
            if not p3_candidates or not p4_candidates or p3_candidates[0] == p4_candidates[0]: continue
            p3, p4 = p3_candidates[0], p4_candidates[0]
            phase_diff_before = np.angle(self.psi[p3] * np.conj(self.psi[p4]))
            phase_diff_after = np.angle(self.psi[p1] * np.conj(self.psi[p2]))
            cost = np.cos(phase_diff_after) - np.cos(phase_diff_before)
            if np.random.rand() < np.exp(3.0 * cost):
                direction = (self.points[p1] - self.points[p2]) + (self.points[p3] - self.points[p4])
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction /= norm
                    force = direction * strength * np.abs(cost)
                    self.velocities[p1] -= force
                    self.velocities[p2] += force

    def update_step(self):
        self._evolve_psi()
        self._try_geometry_flips()
        damping = 0.95
        self.velocities *= damping
        self.points += self.velocities
        self._apply_boundary_conditions()
        self._update_triangulation()

    def close_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

class WavePacketUniverse(Simulation):
    def _initialize_psi(self):
        print("2. Initializing quantum field as a Wave Packet...")
        center = np.array([self.size / 2, self.size / 2])
        distances = np.linalg.norm(self.points - center, axis=1)
        amplitude = np.exp(-distances ** 2 / 2.0)
        momentum_kick = 20.0
        phase = np.exp(1j * self.points[:, 0] * momentum_kick)
        self.psi = amplitude * phase
        self.psi /= np.linalg.norm(self.psi)

class PrimordialSoupUniverse(Simulation):
    def __init__(self, size=40, num_points=500, use_multiprocessing=True, initial_energy=5.0):
        self.initial_energy = initial_energy
        super().__init__(size, num_points, use_multiprocessing)

    def _initialize_psi(self):
        print(f"2. Initializing quantum field from Primordial Soup (Energy = {self.initial_energy})...")
        random_amplitudes = np.random.randn(self.num_points) * self.initial_energy
        random_angles = np.random.uniform(0, 2 * np.pi, self.num_points)
        self.psi = random_amplitudes * np.exp(1j * random_angles)
        self.psi /= np.linalg.norm(self.psi)
