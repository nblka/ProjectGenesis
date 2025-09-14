# renderer.py v18.0
# Part of Project Genesis: Breathing Causality
# v18.0: "Performance First" - A minimalist and high-performance version.
# - The rendering of non-stable "candidate" particles is completely REMOVED
#   from the default render path to ensure maximum speed.
# - All rendering is done with highly optimized, vectorized calls.
# - This is the default, fast renderer for generating long videos.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull
from termcolor import cprint
import os
import traceback

from utils import is_valid_array

worker_substrate_data = {}

def unwrap_numpy_data(d):
    if hasattr(d, 'ndim') and d.ndim == 0:
        return d.item()
    return d

def render_frame_worker(args_tuple):
    frame_num, data_path, frames_dir, shared_info = args_tuple
    try:
        with np.load(data_path, allow_pickle=True) as data:
            psi = unwrap_numpy_data(data['psi'])
            causal_graph = unwrap_numpy_data(data.get('causal_graph'))
            all_tracked_particles = unwrap_numpy_data(data.get('all_tracked_particles'))
            tracked_count = unwrap_numpy_data(data.get('tracked_count', 0))

        substrate_points = worker_substrate_data.get('points')
        substrate_neighbors = worker_substrate_data.get('neighbors')
        if substrate_points is None: raise RuntimeError("Worker not initialized.")

        FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES, DPI = 19.2, 10.8, 100
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
        ax.set_facecolor('#08040E')
        ax.set_aspect('equal')
        points_2d = substrate_points[:, :2]

        # Render STATIC Substrate (fast)
        ax.add_collection(LineCollection(
            [[points_2d[i], points_2d[j]] for i, n_list in enumerate(substrate_neighbors) for j in n_list if i < j],
            colors='#20182D', linewidths=0.6, zorder=1
        ))

        # Render DYNAMIC Causal Graph (fast, vectorized)
        if is_valid_array(causal_graph):
            all_segments, all_colors = [], []; num_segments = 10
            for i, incoming in enumerate(causal_graph):
                if not isinstance(incoming, (list, np.ndarray)): continue
                p_target = points_2d[i]
                for j in incoming:
                    if j < len(points_2d):
                        p_source = points_2d[j]
                        x = np.linspace(p_source[0], p_target[0], num_segments + 1)
                        y = np.linspace(p_source[1], p_target[1], num_segments + 1)
                        pts = np.array([x, y]).T.reshape(-1, 1, 2)
                        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                        all_segments.extend(segs)
                        alphas = np.linspace(0.05, 0.85, num_segments)
                        colors = np.zeros((num_segments, 4)); colors[:,:3] = [0.9,0.9,1.0]; colors[:,3] = alphas
                        all_colors.extend(colors)
            if all_segments:
                ax.add_collection(LineCollection(all_segments, colors=all_colors, linewidths=1.2, zorder=2))

        # Render Nodes using a single `scatter` call (fast)
        amplitudes_sq = np.abs(psi)**2; phases = np.angle(psi)
        hsv_map = plt.get_cmap('hsv'); node_colors = hsv_map((phases + np.pi) / (2 * np.pi))
        max_amp_sq = np.max(amplitudes_sq) if np.max(amplitudes_sq) > 0 else 1.0
        normalized_ampsq = amplitudes_sq / max_amp_sq
        node_sizes = normalized_ampsq * 100 + 8
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=node_sizes, c=node_colors, zorder=3)

        # --- OPTIMIZATION: Render visuals ONLY for STABLE particles ---
        stable_particles = []
        if is_valid_array(all_tracked_particles):
            stable_particles = [p for p in all_tracked_particles if p.get('state') == 'stable']

        if stable_particles:
            # Render Hulls (only for a few particles, so this is fast)
            for particle in stable_particles:
                node_indices = particle.get('node_indices')
                if node_indices is not None and len(node_indices) > 2:
                    try:
                        hull_points = substrate_points[node_indices, :2]
                        hull = ConvexHull(hull_points)
                        for simplex in hull.simplices:
                            ax.plot(hull_points[simplex, 0], hull_points[simplex, 1], color='#00FFFF', linewidth=1.5, zorder=6)
                    except Exception: pass

            # Render Labels (only for a few particles, so this is fast)
            for particle in stable_particles:
                pos_2d = particle["position"][:2]
                label = (f"ID:{particle['track_id']} A:{particle['age']}\n"
                         f"M:{particle['mass']:.3f} Q:{particle['average_charge']:.2f}")
                ax.text(pos_2d[0], pos_2d[1] + 1.0, label, color='#FFFFFF', fontsize=9, ha='center', fontweight='bold', zorder=11, bbox=dict(facecolor='black', alpha=0.7, edgecolor='#00FFFF'))

        # Final Touches and Save
        stable_count = len(stable_particles)
        title = (f"Genesis v18.0 | {shared_info['run_name']}\nFrame: {frame_num} | Tracked: {tracked_count} | Stable: {stable_count}")
        ax.set_title(title, fontsize=14, color='white', pad=20)

        x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
        y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
        ax.set_xlim(x_min - 1, x_max + 1); ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0)

        frame_filename = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
        fig.savefig(frame_filename, dpi=DPI, facecolor=ax.get_facecolor())
        plt.close(fig)
        return None

    except Exception as e:
        return f"Frame {frame_num}: Error - {type(e).__name__} - {e}\n{traceback.format_exc()}"
