# renderer_worker.py v22.4
# Part of Project Genesis: Breathing Causality
# v22.4: "Particle Restoration" - CRITICAL bugfix.
# - Restored the missing rendering logic for stable particles (Layer 4).
# - The worker now correctly draws the Convex Hull and ID label for each
#   stable particle found by the tracker.
# - This fixes the issue where particles were tracked but not visible.

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server/multiprocessing
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import os
import traceback

# --- Global dictionaries for worker process caching ---
worker_substrate_data = {}
worker_colormap_data = {}
worker_log_norm_data = {}

def init_worker(points, neighbors, global_max_amp_sq):
    """Initializes each worker process with the large, static substrate data."""
    worker_substrate_data['points'] = points
    worker_substrate_data['neighbors'] = neighbors

    # --- Pre-calculate log normalization constants ---
    epsilon = 1e-3
    log_max = np.log1p(global_max_amp_sq / epsilon)

    worker_log_norm_data['epsilon'] = epsilon
    worker_log_norm_data['log_max'] = log_max

def unwrap_numpy_data(d):
    """Safely unwraps 0-dimensional numpy arrays to native Python scalars."""
    if hasattr(d, 'ndim') and d.ndim == 0:
        return d.item()
    return d

def is_valid_array(arr):
    """Robustly checks if an object is a non-empty list or numpy array."""
    if isinstance(arr, np.ndarray):
        return arr.size > 0
    if isinstance(arr, (list, tuple)):
        return len(arr) > 0
    return False

def create_heatmap_and_extent(points_2d, values, resolution_h, resolution_w):
    """Creates a heatmap image using robust grid interpolation."""
    x_coords, y_coords = points_2d[:, 0], points_2d[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    padding_x = (x_max - x_min) * 0.02
    padding_y = (y_max - y_min) * 0.02
    extent = [x_min - padding_x, x_max + padding_x, y_min - padding_y, y_max + padding_y]

    grid_y, grid_x = np.mgrid[extent[2]:extent[3]:complex(resolution_h),
                              extent[0]:extent[1]:complex(resolution_w)]

    heatmap = griddata(points_2d, values, (grid_x, grid_y), method='linear', fill_value=0)
    return heatmap, extent

def get_colormaps():
    """Creates and caches custom colormaps for the worker process."""
    if 'heatmap_cmap' in worker_colormap_data:
        return worker_colormap_data['heatmap_cmap'], worker_colormap_data['phase_cmap']

    colors = ["#08040E", "#0c0a2b", "#4a0b5e", "#9b1d5f", "#e2534b", "#fcae1e", "#f0f0c0"]
    nodes = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
    heatmap_cmap = LinearSegmentedColormap.from_list("genesis_heatmap", list(zip(nodes, colors)))
    phase_cmap = plt.get_cmap('hsv')

    worker_colormap_data['heatmap_cmap'] = heatmap_cmap
    worker_colormap_data['phase_cmap'] = phase_cmap
    return heatmap_cmap, phase_cmap


def render_frame_worker(args_tuple):
    """Main worker function to render a single simulation frame."""
    frame_num, data_path, frames_dir, shared_info = args_tuple
    try:
        # --- 1. LOAD DATA ---
        with np.load(data_path, allow_pickle=True) as data:
            psi = unwrap_numpy_data(data['psi'])
            causal_graph = unwrap_numpy_data(data.get('causal_graph'))
            all_tracked_particles = unwrap_numpy_data(data.get('all_tracked_particles'))
            tracked_count = unwrap_numpy_data(data.get('tracked_count', 0))

        # --- 2. LOAD CACHED DATA FROM WORKER ---
        substrate_points = worker_substrate_data.get('points')
        substrate_neighbors = worker_substrate_data.get('neighbors')
        epsilon = worker_log_norm_data.get('epsilon')
        log_max = worker_log_norm_data.get('log_max')
        if substrate_points is None or epsilon is None:
            raise RuntimeError("Worker not initialized correctly.")

        # --- 3. SETUP SCENE ---
        FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES, DPI = 19.2, 10.8, 100
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
        fig.set_facecolor('#08040E')
        ax.set_facecolor('#08040E')
        ax.set_aspect('equal')
        points_2d = substrate_points[:, :2]
        heatmap_cmap, phase_cmap = get_colormaps()

        # --- 4. RENDER LAYERS (BOTTOM TO TOP) ---

        # Layer 0: Energy Heatmap (|psi|^2)
        amplitudes_sq = np.abs(psi)**2
        log_amplitudes = np.log1p(amplitudes_sq / epsilon)
        heatmap, extent = create_heatmap_and_extent(points_2d, log_amplitudes, resolution_h=540, resolution_w=960)
        normalized_heatmap = heatmap / log_max if log_max > 1e-9 else heatmap
        ax.imshow(normalized_heatmap, extent=extent, origin='lower', cmap=heatmap_cmap, zorder=0, aspect='auto', vmin=0.0, vmax=1.0)

        # Layer 1: Static Substrate (Edges)
        ax.add_collection(LineCollection(
            [[points_2d[i], points_2d[j]] for i, n_list in enumerate(substrate_neighbors) for j in n_list if i < j],
            colors='#20182D', linewidths=0.5, zorder=1
        ))

        # Layer 2: Substrate Nodes (fixed size, colored by phase)
        phases = np.angle(psi)
        node_colors = phase_cmap((phases + np.pi) / (2 * np.pi))
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=8, c=node_colors, zorder=3, edgecolors='face', alpha=0.8)

        # Layer 3: Dynamic Causal Graph (Gradient Lines)
        if is_valid_array(causal_graph):
            # (Logic for drawing causal graph remains unchanged)
            source_indices = np.array([j for i, inc in enumerate(causal_graph) for j in inc if is_valid_array(inc) and j < len(points_2d)])
            target_indices = np.array([i for i, inc in enumerate(causal_graph) for _ in inc if is_valid_array(inc)])
            if source_indices.size > 0:
                p_sources = points_2d[source_indices]
                p_targets = points_2d[target_indices]
                p_midpoints = (p_sources + p_targets) / 2.0
                dark_segments = np.array(list(zip(p_sources, p_midpoints)))
                light_segments = np.array(list(zip(p_midpoints, p_targets)))
                ax.add_collection(LineCollection(dark_segments, colors='#303030', linewidths=0.7, zorder=2))
                ax.add_collection(LineCollection(light_segments, colors='#808080', linewidths=0.7, zorder=2))

        # --- FIX: Layer 4: Stable Particles (Hulls and Text) ---
        # This is the block that was missing.
        stable_particles = [p for p in all_tracked_particles if is_valid_array(all_tracked_particles) and p.get('state') == 'stable']
        if stable_particles:
            hulls = []
            for particle in stable_particles:
                node_indices = particle.get('node_indices')
                if is_valid_array(node_indices) and len(node_indices) >= 3:
                    # Get the 2D coordinates for the particle's nodes
                    hull_points = points_2d[node_indices]
                    try:
                        # Calculate the convex hull to find the boundary
                        hull = ConvexHull(hull_points)
                        # Create a polygon patch from the hull vertices
                        polygon = Polygon(hull_points[hull.vertices], closed=True)
                        hulls.append(polygon)

                        # Add the particle ID text at its center of mass
                        pos = particle.get('position')
                        track_id = particle.get('track_id', '?')
                        ax.text(pos[0], pos[1] + 1.5, f"ID:{track_id}",
                                color='white', fontsize=10, ha='center',
                                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
                    except Exception:
                        # This can happen if points are collinear, etc. Fail gracefully.
                        pass
            for particle in stable_particles:
                pos_2d = particle["position"][:2]
                label = (f"ID:{particle['track_id']} A:{particle['age']}\n"
                         f"M:{particle['mass']:.3f} Q:{particle['average_charge']:.2f}")
                ax.text(pos_2d[0], pos_2d[1] + 1.2, label, color='#FFFFFF', fontsize=9, ha='center', alpha=0.7,
                        fontweight='bold', zorder=11, bbox=dict(facecolor='black', alpha=0.5, edgecolor='#FFD700'))

            # Add all hulls to the plot at once for efficiency
            if hulls:
                ax.add_collection(PatchCollection(
                    hulls,
                    facecolor='#FFD700',  # Gold color
                    # edgecolor='#FFFFFF',  # White border
                    edgecolor='#FFD700',  # Gold color
                    linewidth=1.5,
                    linestyles='--',
                    alpha=0.25,            # Semi-transparent
                    zorder=4              # Render on top of most things
                ))
        # --- END OF FIX ---

        # --- 5. FINALIZE AND SAVE ---
        stable_count = len(stable_particles)
        title = (f"Genesis v22.4 | {shared_info.get('run_name', '...')} | "
                 f"Frame: {frame_num} | Tracked: {tracked_count} | Stable: {stable_count}")
        ax.set_title(title, fontsize=14, color='white', pad=20)

        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(pad=0)

        frame_filename = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
        fig.savefig(frame_filename, dpi=DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        return None

    except Exception as e:
        return f"Frame {frame_num}: Error - {type(e).__name__} - {e}\n{traceback.format_exc()}"
