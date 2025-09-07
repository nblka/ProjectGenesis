# tracker.py
import numpy as np

def detect_in_chunk(candidates_chunk, psi, points, vertex_neighbors, phase_threshold):
    """
    Вспомогательная функция для параллельного тестирования кандидатов на 'вихрь' фазы.
    (Этот низкоуровневый код не требует изменений).
    """
    detected_in_chunk = []
    for idx in candidates_chunk:
        neighbors = vertex_neighbors[idx]
        if not neighbors:
            continue

        # Векторное представление центральной фазы
        center_phase_vec = np.array([np.cos(np.angle(psi[idx])), np.sin(np.angle(psi[idx]))])

        # Средний вектор фаз соседей
        neighbor_phases_vecs = np.array([[np.cos(np.angle(psi[n])), np.sin(np.angle(psi[n]))] for n in neighbors])
        avg_neighbor_vec = np.mean(neighbor_phases_vecs, axis=0)

        # 'Кривизна' - это расстояние между центральным вектором и средним вектором соседей.
        # Большое значение означает, что фаза в центре сильно отличается от окружения.
        phase_curvature = np.linalg.norm(center_phase_vec - avg_neighbor_vec)

        if phase_curvature > phase_threshold:
            # Вычисляем 'топологический заряд' как сумму разностей углов
            charge = np.sum(np.angle(psi[neighbors] * np.conj(psi[idx])))
            detected_in_chunk.append({
                "id": idx,
                "charge": charge,
                "position": points[idx],
                "neighbors": neighbors
            })
    return detected_in_chunk


class ParticleTracker:
    """
    Класс-анализатор. Находит и отслеживает стабильные частицы в симуляции.
    """
    def __init__(self, stability_threshold=50, matching_radius=3.0,
                 local_amp_factor=2.0, neighborhood_amp_factor=4.0,
                 phase_threshold=1.5, log_death_threshold=20):

        # --- Параметры для настройки трекера ---
        self.stability_threshold = stability_threshold  # Сколько кадров нужно прожить, чтобы стать 'подтверждаемым'
        self.matching_radius = matching_radius          # Максимальное расстояние для сопоставления частицы между кадрами
        self.local_amp_factor = local_amp_factor        # Во сколько раз амплитуда должна быть выше, чем у прямых соседей
        self.neighborhood_amp_factor = neighborhood_amp_factor # Во сколько раз выше, чем у соседей в радиусе 2 шагов
        self.phase_threshold = phase_threshold          # Порог 'закрученности' фазы для детекции
        self.log_death_threshold = log_death_threshold  # Минимальный возраст исчезнувшей частицы, чтобы сообщить о ней

        # --- Внутреннее состояние трекера ---
        self.tracked_particles = {} # Словарь для хранения отслеживаемых частиц
        self.next_track_id = 0      # Счетчик для уникальных ID

    def _get_neighborhood(self, vertex_neighbors, start_node, depth=2):
        """Рекурсивно находит всех соседей в пределах `depth` шагов."""
        neighborhood = {start_node}
        current_layer = {start_node}
        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                if node < len(vertex_neighbors):
                    for neighbor in vertex_neighbors[node]:
                        next_layer.add(neighbor)
            neighborhood.update(next_layer)
            current_layer = next_layer
        return list(neighborhood)

    def _detect_candidates(self, sim):
        """Основная функция детектирования. Применяет трехступенчатый фильтр."""
        psi = sim.psi
        points = sim.topology.points
        num_points = sim.topology.num_points
        vertex_neighbors = sim.topology.neighbors

        amplitudes_sq = np.abs(psi)**2
        global_mean_amp_sq = np.mean(amplitudes_sq)
        if global_mean_amp_sq < 1e-9: return []

        # --- Фильтр 1 и 2: Двойной порог по амплитуде ---
        pre_candidates = []
        for idx in range(num_points):
            neighbors = vertex_neighbors[idx]
            if len(neighbors) < 2: continue

            # 1. Локальный пик: выше ли амплитуда, чем у прямых соседей?
            mean_neighbor_amp_sq = np.mean(amplitudes_sq[neighbors])
            if amplitudes_sq[idx] > self.local_amp_factor * mean_neighbor_amp_sq:
                 # 2. Региональный пик: выше ли, чем в более широкой окрестности?
                wider_neighborhood = self._get_neighborhood(vertex_neighbors, idx, depth=2)
                mean_wider_amp_sq = np.mean(amplitudes_sq[wider_neighborhood])
                if amplitudes_sq[idx] > self.neighborhood_amp_factor * mean_wider_amp_sq:
                     pre_candidates.append(idx)

        if not pre_candidates:
            return []

        # --- Фильтр 3: Кривизна фазы (поиск 'вихрей') ---
        # (Используем вспомогательную функцию, которую можно распараллелить в будущем)
        phase_passed = detect_in_chunk(pre_candidates, psi, points, vertex_neighbors, self.phase_threshold)

        # Вычисляем эмерджентную 'массу' для прошедших кандидатов
        for candidate in phase_passed:
            candidate['mass'] = amplitudes_sq[candidate['id']] / global_mean_amp_sq
        return phase_passed

    def analyze_frame(self, sim, frame_num):
        """
        Главный метод, вызываемый из основного цикла.
        Анализирует один кадр, обновляет состояние всех отслеживаемых частиц.
        """
        current_candidates = self._detect_candidates(sim)
        unmatched_candidates = list(current_candidates)

        # --- Шаг 1: Сопоставление существующих треков с новыми кандидатами ---
        for track_id, particle_data in self.tracked_particles.items():
            best_match, min_dist = None, self.matching_radius
            for candidate in unmatched_candidates:
                dist = np.linalg.norm(particle_data["position"] - candidate["position"])
                if dist < min_dist:
                    min_dist, best_match = dist, candidate

            if best_match:
                # Обновляем данные трека
                particle_data.update({k: v for k, v in best_match.items() if k != "charge"})
                particle_data["charge_history"].append(best_match["charge"])
                particle_data["average_charge"] = np.mean(particle_data["charge_history"])
                particle_data["age"] += 1
                particle_data["last_seen_frame"] = frame_num
                unmatched_candidates.remove(best_match)

        # --- Шаг 2: Удаление 'потерянных' треков ---
        dead_tracks = [track_id for track_id, p_data in self.tracked_particles.items() if p_data["last_seen_frame"] < frame_num]
        for track_id in dead_tracks:
            p_data = self.tracked_particles[track_id]
            # Сообщаем об исчезновении только если частица была достаточно 'старой'
            if p_data['age'] > self.log_death_threshold:
                print(f"--- Particle Track ID: {track_id} (State: {p_data['state']}) has disappeared! Lived for {p_data['age']} frames.")
            del self.tracked_particles[track_id]

        # --- Шаг 3: Создание новых треков для 'новичков' ---
        for candidate in unmatched_candidates:
            new_id = self.next_track_id
            self.next_track_id += 1
            candidate["charge_history"] = [candidate["charge"]]
            candidate["average_charge"] = candidate["charge"]
            self.tracked_particles[new_id] = {
                **candidate,
                "age": 1,
                "last_seen_frame": frame_num,
                "state": "tracking", # Начальное состояние - просто отслеживается
                "confirmation_age": 0
            }

        # --- Шаг 4: Проверка на стабильность и обновление состояний ---
        stable_particles = []
        for track_id, p_data in self.tracked_particles.items():
            # Переход из 'отслеживается' в 'подтверждается'
            if p_data["state"] == "tracking" and p_data["age"] >= self.stability_threshold:
                p_data["state"] = "confirming"
                p_data["confirmation_age"] = 1
                print(f"--- Particle ID: {track_id} entering CONFIRMING state at age {p_data['age']} ---")

            # Переход из 'подтверждается' в 'стабильная'
            elif p_data["state"] == "confirming":
                p_data["confirmation_age"] += 1
                if p_data["confirmation_age"] >= self.stability_threshold / 2: # Подтверждение требует меньше времени
                    p_data["state"] = "stable"
                    print(f"--- PARTICLE CONFIRMED STABLE! ID: {track_id}, Age: {p_data['age']}, M: {p_data['mass']:.2f}, Q: {p_data['average_charge']:.2f} ---")

            if p_data["state"] == "stable":
                p_data['track_id'] = track_id
                stable_particles.append(p_data)

        return stable_particles
