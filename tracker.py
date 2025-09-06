# tracker.py v2.2
"""
Particle Tracking Module for Project Genesis v2.2 (Robust Detection)
---------------------------------------------------------------------
- Implements dual-local threshold detection (local peak + wider neighborhood).
- Robust stability logic with 'confirming' state.
"""
import numpy as np

def detect_in_chunk(candidates_chunk, psi, points, vertex_neighbors, phase_threshold):
    detected_in_chunk = []
    for idx in candidates_chunk:
        neighbors = vertex_neighbors[idx]
        if not neighbors: continue
        center_phase_vec = np.array([np.cos(np.angle(psi[idx])), np.sin(np.angle(psi[idx]))])
        neighbor_phases_vecs = np.array([[np.cos(np.angle(psi[n])), np.sin(np.angle(psi[n]))] for n in neighbors])
        avg_neighbor_vec = np.mean(neighbor_phases_vecs, axis=0)
        phase_curvature = np.linalg.norm(center_phase_vec - avg_neighbor_vec)
        if phase_curvature > phase_threshold:
            charge = np.sum(np.angle(psi[neighbors] * np.conj(psi[idx])))
            detected_in_chunk.append({"id": idx, "charge": charge, "position": points[idx], "neighbors": neighbors})
    return detected_in_chunk

class ParticleTracker:
    def __init__(self, stability_threshold=100, matching_radius=3.0,
                 local_amp_factor=1.5, neighborhood_amp_factor=3.0,
                 phase_threshold=1.8, log_death_threshold=20, pool=None):
        self.stability_threshold = stability_threshold
        self.matching_radius = matching_radius
        self.local_amp_factor = local_amp_factor
        self.neighborhood_amp_factor = neighborhood_amp_factor
        self.phase_threshold = phase_threshold
        self.log_death_threshold = log_death_threshold
        self.pool = pool
        self.tracked_particles = {}
        self.next_track_id = 0

    def _get_neighborhood(self, sim, start_node, depth=2):
        neighborhood = {start_node}
        current_layer = {start_node}
        for _ in range(depth):
            next_layer = set()
            for node in current_layer:
                for neighbor in sim.vertex_neighbors[node]: next_layer.add(neighbor)
            neighborhood.update(next_layer)
            current_layer = next_layer
        return list(neighborhood)

    def _detect_candidates(self, sim):
        amplitudes_sq = np.abs(sim.psi)**2
        global_mean_amp_sq = np.mean(amplitudes_sq)

        # We check ALL nodes
        pre_candidates = []
        for idx in range(sim.num_points):
            neighbors = sim.vertex_neighbors[idx]
            if len(neighbors) < 2: continue

            # 1. LOCAL PEAK TEST vs immediate neighbors
            mean_neighbor_amp_sq = np.mean(amplitudes_sq[neighbors])
            if amplitudes_sq[idx] > self.local_amp_factor * mean_neighbor_amp_sq:
                 # 2. WIDER NEIGHBORHOOD TEST vs broader area
                wider_neighborhood = self._get_neighborhood(sim, idx, depth=2)
                mean_wider_amp_sq = np.mean(amplitudes_sq[wider_neighborhood])
                if amplitudes_sq[idx] > self.neighborhood_amp_factor * mean_wider_amp_sq:
                     pre_candidates.append(idx)

        if not pre_candidates: return []

        # 3. Phase Curvature Test
        phase_passed = detect_in_chunk(pre_candidates, sim.psi, sim.points, sim.vertex_neighbors, self.phase_threshold)

        for candidate in phase_passed:
            candidate['mass'] = amplitudes_sq[candidate['id']] / global_mean_amp_sq
        return phase_passed

    def analyze_frame(self, sim, frame_num):
        # ... (вся остальная логика отслеживания, подтверждения и удаления остается прежней)
        current_candidates = self._detect_candidates(sim)
        unmatched_candidates = list(current_candidates)
        for track_id, particle_data in self.tracked_particles.items():
            best_match, min_dist = None, self.matching_radius
            for candidate in unmatched_candidates:
                dist = np.linalg.norm(particle_data["position"] - candidate["position"])
                if dist < min_dist:
                    min_dist, best_match = dist, candidate
            if best_match:
                particle_data["charge_history"].append(best_match["charge"])
                particle_data["average_charge"] = np.mean(particle_data["charge_history"])
                particle_data.update({k: v for k, v in best_match.items() if k != "charge"})
                particle_data["age"] += 1
                particle_data["last_seen_frame"] = frame_num
                unmatched_candidates.remove(best_match)
        dead_tracks = [track_id for track_id, p_data in self.tracked_particles.items() if p_data["last_seen_frame"] < frame_num]
        for track_id in dead_tracks:
            p_data = self.tracked_particles[track_id]
            if p_data['age'] > self.log_death_threshold:
                print(f"--- Particle Track ID: {track_id} (State: {p_data['state']}) has disappeared! Lived for {p_data['age']} frames.")
            del self.tracked_particles[track_id]
        for candidate in unmatched_candidates:
            new_id = self.next_track_id
            candidate["charge_history"] = [candidate["charge"]]
            candidate["average_charge"] = candidate["charge"]
            self.tracked_particles[new_id] = {**candidate, "age": 1, "last_seen_frame": frame_num, "state": "tracking", "confirmation_age": 0}
            self.next_track_id += 1
        stable_particles = []
        for track_id, p_data in self.tracked_particles.items():
            if p_data["state"] == "tracking" and p_data["age"] >= self.stability_threshold:
                p_data["state"] = "confirming"
                p_data["confirmation_age"] = 1
                print(f"--- Particle ID: {track_id} entering CONFIRMING state at age {p_data['age']} ---")
            elif p_data["state"] == "confirming":
                p_data["confirmation_age"] += 1
                if p_data["confirmation_age"] >= self.stability_threshold:
                    p_data["state"] = "stable"
                    print(f"--- PARTICLE CONFIRMED STABLE! ID: {track_id}, Total Age: {p_data['age']}, Avg Charge: {p_data['average_charge']:.3f} ---")
            if p_data["state"] == "stable":
                p_data['track_id'] = track_id
                stable_particles.append(p_data)
        return stable_particles
