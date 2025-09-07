# topologies.py
import numpy as np
from tqdm import tqdm
from termcolor import cprint

class CrystalTopology:
    def __init__(self, width=80, height=60):
        cprint(f"1. Building Topology via Genesis Algorithm ({width}x{height})...", 'cyan', attrs=['bold'])
        self.width = width
        self.height = height
        self.num_points = width * height

        self.points = np.zeros((self.num_points, 2), dtype=float)
        self.neighbors = [[] for _ in range(self.num_points)]

        # --- АЛГОРИТМ "ГЕНЕЗИС" ---
        for r in range(height): # Идем ряд за рядом (r = row)
            for q in range(width):  # Идем узел за узлом в ряду (q = column)

                idx = r * width + q

                # 1. Размещаем узел в пространстве (визуализация)
                # Сдвигаем только нечетные ряды для 'зигзага'
                px = q + 0.5 * (r % 2)
                py = r * np.sqrt(3) / 2
                self.points[idx] = [px, py]

                # 2. Определяем соседей на основе уже 'построенной' части мира
                # Сосед слева
                if q > 0:
                    left_idx = r * width + (q - 1)
                    self.neighbors[idx].append(left_idx)
                    self.neighbors[left_idx].append(idx)

                # Соседи сверху (их положение зависит от четности ряда)
                if r > 0:
                    if r % 2 == 0: # Четный ряд
                        # Соседи: верхний-левый и верхний-правый
                        if q > 0:
                            top_left_idx = (r - 1) * width + (q - 1)
                            self.neighbors[idx].append(top_left_idx)
                            self.neighbors[top_left_idx].append(idx)
                        top_right_idx = (r - 1) * width + q
                        self.neighbors[idx].append(top_right_idx)
                        self.neighbors[top_right_idx].append(idx)
                    else: # Нечетный ряд
                        # Соседи: верхний-левый и верхний-правый
                        top_left_idx = (r - 1) * width + q
                        self.neighbors[idx].append(top_left_idx)
                        self.neighbors[top_left_idx].append(idx)
                        if q < width - 1:
                            top_right_idx = (r - 1) * width + (q + 1)
                            self.neighbors[idx].append(top_right_idx)
                            self.neighbors[top_right_idx].append(idx)

        self.points -= np.mean(self.points, axis=0)

    def get_simplices(self):
        print("   -> Generating simplices (bulletproof method on Genesis lattice)...")
        simplices_set = set()
        neighbor_sets = [set(n) for n in self.neighbors]
        for idx1 in tqdm(range(self.num_points), desc="   - Finding Triangles"):
            for idx2 in self.neighbors[idx1]:
                if idx1 > idx2: continue
                common_neighbors = neighbor_sets[idx1].intersection(neighbor_sets[idx2])
                for idx3 in common_neighbors:
                    if idx2 > idx3: continue
                    simplices_set.add((idx1, idx2, idx3))
        return np.array(list(simplices_set), dtype=int)
