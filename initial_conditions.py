# initial_conditions.py
import numpy as np

class BaseInitialState:
    def generate(self, topology):
        raise NotImplementedError

class WavePacketState(BaseInitialState):
    def __init__(self, momentum_kick=30.0):
        print(f"   -> IC: Coherent Wave Packet (kick={momentum_kick})")
        self.momentum_kick = momentum_kick

    def generate(self, topology):
        center = np.mean(topology.points, axis=0)
        distances = np.linalg.norm(topology.points - center, axis=1)
        # Делаем пакет более узким
        amplitude = np.exp(-distances ** 2 / (2.0 * (topology.width / 10)**2))
        phase = np.exp(1j * topology.points[:, 0] * self.momentum_kick)
        psi = amplitude * phase
        return psi / np.linalg.norm(psi)

class PrimordialSoupState(BaseInitialState):
    def __init__(self):
        print(f"   -> IC: Primordial Soup")

    def generate(self, topology):
        random_amplitudes = np.random.randn(topology.num_points)
        random_angles = np.random.uniform(0, 2 * np.pi, topology.num_points)
        psi = random_amplitudes * np.exp(1j * random_angles)
        return psi / np.linalg.norm(psi)


class VortexState(BaseInitialState):
    def __init__(self, position_ratio=(0.5, 0.5), vortex_radius=1):
        print(f"   -> IC: Single Topological Vortex")
        self.position_ratio = position_ratio
        self.vortex_radius = vortex_radius # Пока не используется, но оставим для будущего

    def generate(self, topology):
        # 1. Создаем "холодный вакуум"
        # ИСПРАВЛЕНИЕ: Начинаем с истинного "голубого" вакуума (фаза PI)
        psi = np.ones(topology.num_points, dtype=complex) * np.exp(1j * np.pi)

        # 2. Находим центр для нашего вихря
        # ИСПРАВЛЕНИЕ: Вычисляем геометрический центр всех точек
        min_coords = np.min(topology.points, axis=0)
        max_coords = np.max(topology.points, axis=0)

        # Точка для вихря будет интерполяцией между min и max координатами
        center_point_coords = min_coords + (max_coords - min_coords) * np.array(self.position_ratio)

        # Находим узел, ближайший к вычисленной центральной точке
        distances_to_center = np.linalg.norm(topology.points - center_point_coords, axis=1)
        center_idx = np.argmin(distances_to_center)

        # 3. Создаем вихрь по вашему рецепту
        neighbors_of_center = topology.neighbors[center_idx]

        # Устанавливаем фазу центрального узла на 0 (противофаза для "голубого" вакуума)
        psi[center_idx] = np.exp(1j * 0)

        # 4. Устанавливаем амплитуды
        vortex_nodes = [center_idx] + neighbors_of_center
        amplitudes = np.ones(topology.num_points) * 0.01 # Очень тихий вакуум
        amplitudes[vortex_nodes] = 1.0 # Пик энергии в центре вихря
        psi *= amplitudes

        # 5. Финальная нормализация
        return psi / np.linalg.norm(psi)
