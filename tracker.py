# tracker.py v3.0 - "Quantum Genome"
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label
from termcolor import cprint

class ParticleTracker:
    """
    Класс-анализатор v3.0. Находит и отслеживает протяженные, стабильные
    квазичастицы ('комки'), используя алгоритм оптимального сопоставления
    на основе сохраняющихся 'квантовых чисел'.
    """
    def __init__(self,
                 # --- Параметры для детектирования ---
                 amp_threshold_factor=3.0,     # Порог амплитуды для сегментации
                 min_clump_size=5,             # Минимальный размер 'комка' в узлах

                 # --- Параметры для трекинга (ключевое нововведение) ---
                 matching_cost_threshold=10.0, # Макс. 'стоимость' для сопоставления. Если выше - это рождение/смерть.
                 mass_weight=1.0,              # Вес разницы масс в функции стоимости
                 charge_weight=5.0,            # Вес разницы зарядов (самый важный)
                 distance_weight=0.2,          # Вес пространственного расстояния (менее важный)

                 # --- Параметры для жизненного цикла частиц ---
                 stability_threshold=50,       # Сколько кадров нужно прожить, чтобы стать 'стабильной'
                 log_death_threshold=20):      # Минимальный возраст исчезнувшей частицы для логирования

        # Сохраняем параметры
        self.amp_threshold_factor = amp_threshold_factor
        self.min_clump_size = min_clump_size
        self.matching_cost_threshold = matching_cost_threshold
        self.mass_weight = mass_weight
        self.charge_weight = charge_weight
        self.distance_weight = distance_weight
        self.stability_threshold = stability_threshold
        self.log_death_threshold = log_death_threshold

        # Внутреннее состояние трекера
        self.tracked_particles = {}
        self.next_track_id = 0

    def _detect_and_characterize_clumps(self, sim):
        """ШАГ 1 & 2: Находит и измеряет свойства 'комков' в одном кадре."""
        psi = sim.psi
        points = sim.topology.points
        amplitudes_sq = np.abs(psi)**2

        # --- 1. Сегментация по Амплитуде ---
        global_mean_amp_sq = np.mean(amplitudes_sq)
        if global_mean_amp_sq < 1e-9: return []

        threshold = self.amp_threshold_factor * global_mean_amp_sq
        binary_map = amplitudes_sq > threshold

        # Создаем структуру для поиска соседей на решетке
        structure = np.array([[0,1,0], [1,1,1], [0,1,0]]) # Соседи по ребрам
        labeled_map, num_features = label(binary_map.reshape(sim.topology.height, sim.topology.width), structure=structure)
        labeled_map = labeled_map.flatten()

        if num_features == 0:
            return []

        # --- 2. Характеризация найденных 'комков' ---
        candidates = []
        for i in range(1, num_features + 1):
            # Находим все узлы, принадлежащие этому 'комку'
            node_indices = np.where(labeled_map == i)[0]

            if len(node_indices) < self.min_clump_size:
                continue

            clump_psi = psi[node_indices]
            clump_amps_sq = amplitudes_sq[node_indices]
            clump_points = points[node_indices]

            # Вычисляем эмерджентные свойства
            mass = np.sum(clump_amps_sq)
            position = np.average(clump_points, weights=clump_amps_sq, axis=0)

            # Более точный 'топологический' заряд
            center_phase = np.angle(np.sum(clump_psi * clump_amps_sq)) # Средневзвешенная фаза
            charge = np.sum(np.angle(np.exp(1j * (np.angle(clump_psi) - center_phase)))) / (2 * np.pi)

            candidates.append({
                "mass": mass,
                "charge": charge,
                "position": position,
                "size": len(node_indices),
                "node_indices": node_indices # Сохраняем узлы для будущих нужд
            })
        return candidates

    def _create_new_track(self, candidate, frame_num):
        """Создает запись для новой, ранее не виденной частицы."""
        new_id = self.next_track_id
        self.next_track_id += 1

        self.tracked_particles[new_id] = {
            **candidate,
            "track_id": new_id,
            "age": 1,
            "last_seen_frame": frame_num,
            "state": "tracking",
            "charge_history": [candidate["charge"]],
            "average_charge": candidate["charge"]
        }

    def _update_track(self, track_id, candidate, frame_num):
        """Обновляет данные существующей частицы."""
        particle_data = self.tracked_particles[track_id]

        # Обновляем все, кроме истории
        particle_data.update(candidate)

        particle_data["age"] += 1
        particle_data["last_seen_frame"] = frame_num

        # Обновляем скользящее среднее для заряда
        particle_data["charge_history"].append(candidate["charge"])
        if len(particle_data["charge_history"]) > 10: # Ограничиваем историю
            particle_data["charge_history"].pop(0)
        particle_data["average_charge"] = np.mean(particle_data["charge_history"])

    def analyze_frame(self, sim, frame_num):
        """Главный метод анализа. Вызывается из main.py на каждом шаге."""
        current_candidates = self._detect_and_characterize_clumps(sim)

        # --- Управление жизненным циклом треков ---
        # 1. Отмечаем все существующие треки как 'невиденные' в этом кадре
        for p_data in self.tracked_particles.values():
            p_data['seen_this_frame'] = False

        # 2. Если есть что сопоставлять, строим матрицу стоимости
        previous_tracks = list(self.tracked_particles.values())
        if previous_tracks and current_candidates:
            cost_matrix = np.full((len(previous_tracks), len(current_candidates)), np.inf)

            for i, p_old in enumerate(previous_tracks):
                for j, p_new in enumerate(current_candidates):
                    mass_diff = abs(p_old['mass'] - p_new['mass']) / (p_old['mass'] + 1e-9) # Нормируем
                    charge_diff = abs(p_old['average_charge'] - p_new['charge'])
                    dist_diff = np.linalg.norm(p_old['position'] - p_new['position'])

                    cost = (self.mass_weight * mass_diff +
                            self.charge_weight * charge_diff +
                            self.distance_weight * dist_diff)
                    cost_matrix[i, j] = cost

            # 3. Находим оптимальные пары
            old_indices, new_indices = linear_sum_assignment(cost_matrix)

            # 4. Обновляем сопоставленные треки
            for i, j in zip(old_indices, new_indices):
                if cost_matrix[i, j] < self.matching_cost_threshold:
                    track_id_to_update = previous_tracks[i]['track_id']
                    self._update_track(track_id_to_update, current_candidates[j], frame_num)
                    self.tracked_particles[track_id_to_update]['seen_this_frame'] = True
                    current_candidates[j]['matched'] = True # Помечаем кандидата как использованного

        # 5. Обрабатываем 'смерти' и 'рождения'
        dead_tracks = []
        for track_id, p_data in self.tracked_particles.items():
            if not p_data.get('seen_this_frame', False):
                if p_data['age'] > self.log_death_threshold:
                    cprint(f"INFO: Particle ID {track_id} (age {p_data['age']}) disappeared.", 'yellow')
                dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self.tracked_particles[track_id]

        for candidate in current_candidates:
            if not candidate.get('matched', False):
                self._create_new_track(candidate, frame_num)

        # --- Обновление состояний (стабильность) и возврат списка для рендерера ---
        stable_particles_list = []
        for track_id, p_data in self.tracked_particles.items():
            if p_data['age'] >= self.stability_threshold and p_data.get('state') != 'stable':
                p_data['state'] = 'stable'
                cprint(f"STABLE PARTICLE CONFIRMED! ID: {track_id}, Age: {p_data['age']}, "
                       f"Mass: {p_data['mass']:.2f}, Charge: {p_data['average_charge']:.2f}", 'green')

            if p_data.get('state') == 'stable':
                stable_particles_list.append(p_data)

        return stable_particles_list
