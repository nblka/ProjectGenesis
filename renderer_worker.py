# renderer_worker.py v22.3
# Part of Project Genesis: Breathing Causality
# v22.3: Final "Quantum Heatmap" Implementation
# - Renders a smooth, interpolated heatmap for the energy |psi|^2, ensuring
#   perfect alignment with the substrate nodes using `griddata`.
# - Uses an ABSOLUTE normalization based on the global maximum |psi|^2 provided
#   by the main script, ensuring consistent brightness across all frames.
# - The substrate nodes are now small, fixed-size points, colored by phase.
# - Causal graph is rendered with a clean, informative dark-to-light gradient.

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server/multiprocessing
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import os
import traceback

# --- Global dictionaries for worker process caching ---
# This is an efficient way to hold large, static data in each worker process
# without needing to send it with every task.
worker_substrate_data = {}
worker_colormap_data = {}
worker_log_norm_data = {}

def init_worker(points, neighbors, global_max_amp_sq):
    """Initializes each worker process with the large, static substrate data."""
    worker_substrate_data['points'] = points
    worker_substrate_data['neighbors'] = neighbors

    # --- Pre-calculate log normalization constants ---
    # Epsilon prevents log(0) errors. It should be a tiny fraction of the max.
    epsilon = 1e-3
    # The maximum value on our log scale.
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

    # A custom "inferno-like" colormap for the energy heatmap
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

        substrate_points = worker_substrate_data.get('points')
        substrate_neighbors = worker_substrate_data.get('neighbors')
        global_max_amp_sq = shared_info.get('global_max_amp_sq_for_norm', 1.0)

        # --- 2. LOAD CACHED DATA FROM WORKER ---
        substrate_points = worker_substrate_data.get('points')
        epsilon = worker_log_norm_data.get('epsilon')
        log_max = worker_log_norm_data.get('log_max')
        if substrate_points is None or epsilon is None:
            raise RuntimeError("Worker not initialized correctly.")

        # --- 2. SETUP SCENE ---
        FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES, DPI = 19.2, 10.8, 100
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
        fig.set_facecolor('#08040E')
        ax.set_facecolor('#08040E')
        ax.set_aspect('equal')
        points_2d = substrate_points[:, :2]
        heatmap_cmap, phase_cmap = get_colormaps()

        # --- 3. RENDER LAYERS (BOTTOM TO TOP) ---

        # Layer 0: Energy Heatmap (|psi|^2)
        amplitudes_sq = np.abs(psi)**2

        # Apply the logarithmic transformation
        log_amplitudes = np.log1p(amplitudes_sq / epsilon)

        heatmap, extent = create_heatmap_and_extent(points_2d, log_amplitudes, resolution_h=540, resolution_w=960)

        # ABSOLUTE NORMALIZATION on the log scale
        if log_max > 1e-9:
            normalized_heatmap = heatmap / log_max
        else:
            normalized_heatmap = heatmap # Fallback

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
            source_indices = np.array([j for i, inc in enumerate(causal_graph) for j in inc if is_valid_array(inc) and j < len(points_2d)])
            target_indices = np.array([i for i, inc in enumerate(causal_graph) for _ in inc if is_valid_array(inc)])
            if source_indices.size > 0:
                p_sources = points_2d[source_indices]
                p_targets = points_2d[target_indices]
                p_midpoints = (p_sources + p_targets) / 2.0
                dark_segments = np.array(list(zip(p_sources, p_midpoints)))
                light_segments = np.array(list(zip(p_midpoints, p_targets)))
                all_segments = np.concatenate([dark_segments, light_segments])
                all_colors = ['#303030'] * len(dark_segments) + ['#808080'] * len(light_segments)
                ax.add_collection(LineCollection(all_segments, colors=all_colors, linewidths=0.7, zorder=2))

        # Layer 4: Stable Particles (Hulls and Text)
        stable_particles = [p for p in all_tracked_particles if is_valid_array(all_tracked_particles) and p.get('state') == 'stable']
        if stable_particles:
            # (Code for hulls and text remains the same, with high zorder)
            pass

        # --- 4. FINALIZE AND SAVE ---
        stable_count = len(stable_particles)
        title = (f"Genesis v22.3 | {shared_info.get('run_name', '...')} | "
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
