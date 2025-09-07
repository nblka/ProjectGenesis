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
