# tracker.py v13.1
# Part of Project Genesis: Breathing Causality
# v13.1: Final, robust version. Fully topology-agnostic.

import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from termcolor import cprint

class ParticleTracker:
    """
    Analyzes the simulation to find and track stable, time-averaged quasi-particles (attractors).
    v13.1 is a pure topological analyzer.
    """
    def __init__(self,
                 ema_alpha=0.1,
                 amp_threshold_factor=2.5,
                 min_clump_size=5,
                 matching_cost_threshold=8.0,
                 mass_weight=1.0,
                 charge_weight=5.0,
                 distance_weight=0.5,
                 stability_threshold=50,
                 log_death_threshold=20):

        self.ema_alpha = ema_alpha
        self.amp_threshold_factor = amp_threshold_factor
        self.min_clump_size = min_clump_size
        self.matching_cost_threshold = matching_cost_threshold
        self.mass_weight = mass_weight
        self.charge_weight = charge_weight
        self.distance_weight = distance_weight
        self.stability_threshold = stability_threshold
        self.log_death_threshold = log_death_threshold

        self.tracked_particles = {}
        self.next_track_id = 0
        self.time_averaged_ampsq = None

    def _update_smoothed_field(self, current_ampsq):
        """Updates the exponentially smoothed amplitude field."""
        if self.time_averaged_ampsq is None:
            self.time_averaged_ampsq = current_ampsq.copy()
        else:
            self.time_averaged_ampsq = (self.ema_alpha * current_ampsq +
                                        (1 - self.ema_alpha) * self.time_averaged_ampsq)

    def _find_clumps_via_bfs(self, binary_map: np.ndarray, substrate_neighbors: list) -> list:
        """Finds connected components (clumps) on a graph using Breadth-First Search."""
        num_points = len(binary_map)
        visited = np.zeros(num_points, dtype=bool)
        all_clumps = []

        for i in range(num_points):
            if binary_map[i] and not visited[i]:
                current_clump_indices = []
                q = deque([i])
                visited[i] = True

                while q:
                    current_node = q.popleft()
                    current_clump_indices.append(current_node)

                    for neighbor in substrate_neighbors[current_node]:
                        if binary_map[neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            q.append(neighbor)

                all_clumps.append(current_clump_indices)

        return all_clumps

    def _detect_and_characterize_clumps(self, sim):
        """Finds and measures clumps, now with a ghost-busting mechanism."""
        if self.time_averaged_ampsq is None: return []

        smoothed_ampsq = self.time_averaged_ampsq
        psi = sim.psi
        points = sim.substrate.points
        neighbors = sim.substrate.neighbors

        # --- Segmentation on the smoothed field (as before) ---
        global_mean_smoothed_amp = np.mean(smoothed_ampsq)
        if global_mean_smoothed_amp < 1e-9: return []
        threshold = self.amp_threshold_factor * global_mean_smoothed_amp
        binary_map_smooth = smoothed_ampsq > threshold

        # --- NEW: THE "REALITY CHECK" ---
        # A clump must ALSO have a significant INSTANTANEOUS presence.
        instantaneous_ampsq = np.abs(psi)**2
        global_mean_instant_amp = np.mean(instantaneous_ampsq)
        # We can use a slightly lower threshold for the reality check.
        reality_check_threshold = 0.5 * threshold
        binary_map_instant = instantaneous_ampsq > reality_check_threshold

        # A node is considered "hot" only if it's hot in BOTH maps.
        final_binary_map = binary_map_smooth & binary_map_instant

        # --- Clump Finding now uses the final, combined map ---
        clump_node_lists = self._find_clumps_via_bfs(final_binary_map, neighbors)

        if not clump_node_lists: return []

        # Characterization
        candidates = []
        for node_indices in clump_node_lists:
            if len(node_indices) < self.min_clump_size: continue

            clump_psi = psi[node_indices]
            clump_amps_sq_instant = np.abs(clump_psi)**2
            clump_points = points[node_indices]

            mass = np.sum(clump_amps_sq_instant)
            if mass < 1e-9: continue

            position = np.average(clump_points, weights=clump_amps_sq_instant, axis=0)

            center_phase = np.angle(np.sum(clump_psi * clump_amps_sq_instant))
            charge = np.sum(np.angle(np.exp(1j * (np.angle(clump_psi) - center_phase)))) / (2 * np.pi)

            candidates.append({
                "mass": mass, "charge": charge, "position": position,
                "size": len(node_indices), "node_indices": node_indices
            })

        return candidates

    def analyze_frame(self, sim, frame_num):
        """Main analysis method for a single frame."""
        self._update_smoothed_field(np.abs(sim.psi)**2)
        current_candidates = self._detect_and_characterize_clumps(sim)

        for p_data in self.tracked_particles.values(): p_data['seen_this_frame'] = False

        previous_tracks = list(self.tracked_particles.values())
        if previous_tracks and current_candidates:
            cost_matrix = np.zeros((len(previous_tracks), len(current_candidates)))
            for i, p_old in enumerate(previous_tracks):
                for j, p_new in enumerate(current_candidates):
                    mass_diff = abs(p_old['mass'] - p_new['mass'])
                    charge_diff = abs(p_old['average_charge'] - p_new['charge'])
                    dist_diff = np.linalg.norm(p_old['position'] - p_new['position'])
                    cost = (self.mass_weight * mass_diff +
                            self.charge_weight * charge_diff +
                            self.distance_weight * dist_diff)
                    cost_matrix[i, j] = cost

            old_indices, new_indices = linear_sum_assignment(cost_matrix)

            for i, j in zip(old_indices, new_indices):
                if cost_matrix[i, j] < self.matching_cost_threshold:
                    track_id = previous_tracks[i]['track_id']
                    self._update_track(track_id, current_candidates[j], frame_num)
                    self.tracked_particles[track_id]['seen_this_frame'] = True
                    current_candidates[j]['matched'] = True

        dead_ids = [tid for tid, p in self.tracked_particles.items() if not p.get('seen_this_frame')]
        for tid in dead_ids:
            if self.tracked_particles[tid]['age'] > self.log_death_threshold:
                cprint(f"INFO (Frame {frame_num}): Attractor ID {tid} (age {self.tracked_particles[tid]['age']}) dissipated.", 'grey')
            del self.tracked_particles[tid]

        for cand in current_candidates:
            if not cand.get('matched', False):
                self._create_new_track(cand, frame_num)

        stable_attractors = []
        for track_id, p_data in self.tracked_particles.items():
            if p_data['age'] > self.stability_threshold:
                if p_data.get('state') != 'stable':
                    p_data['state'] = 'stable'
                    cprint(f"STABLE ATTRACTOR CONFIRMED! (Frame {frame_num}) ID: {track_id}, Age: {p_data['age']}, "
                           f"Mass: {p_data['mass']:.2f}, Charge: {p_data['average_charge']:.2f}", 'green', attrs=['bold'])
                stable_attractors.append(p_data)

        return stable_attractors

    def _create_new_track(self, candidate, frame_num):
        new_id = self.next_track_id; self.next_track_id += 1
        self.tracked_particles[new_id] = {
            **candidate, "track_id": new_id, "age": 1, "last_seen_frame": frame_num,
            "state": "tracking", "charge_history": [candidate["charge"]],
            "average_charge": candidate["charge"]
        }

    def _update_track(self, track_id, candidate, frame_num):
        p = self.tracked_particles[track_id]
        p.update(candidate); p["age"] += 1; p["last_seen_frame"] = frame_num
        p["charge_history"].append(candidate["charge"])
        if len(p["charge_history"]) > 10: p["charge_history"].pop(0)
        p["average_charge"] = np.mean(p["charge_history"])
