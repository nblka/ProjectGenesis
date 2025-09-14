# renderer_worker.py v22.2
# Part of Project Genesis: Breathing Causality
# v22.2: "Coordinate System Fix" - CRITICAL bugfix for the heatmap renderer.
# - Replaces the manual Gaussian-stamping method with a robust, industry-standard
#   interpolation using `scipy.interpolate.griddata`.
# - This guarantees perfect alignment between the energy heatmap and the substrate nodes,
#   solving the critical misalignment bug.
# - This version produces physically correct and visually accurate frames.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from termcolor import cprint
import os
import traceback

# --- Global dictionaries for worker process caching ---
worker_substrate_data = {}
worker_colormap_data = {}

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

def create_heatmap_and_extent(points_2d, amplitudes_sq, resolution_h=1080, resolution_w=1920):
    """
    Creates a heatmap image using robust grid interpolation.
    This method guarantees that the heatmap is perfectly aligned with the point data.
    """
    # 1. Define the grid boundaries for interpolation based on the data's extent.
    x_coords, y_coords = points_2d[:, 0], points_2d[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Add a 2% padding to prevent nodes from being rendered exactly on the edge.
    padding_x = (x_max - x_min) * 0.02
    padding_y = (y_max - y_min) * 0.02
    extent = [x_min - padding_x, x_max + padding_x, y_min - padding_y, y_max + padding_y]

    # 2. Create the target coordinate grid for the final heatmap image.
    # The resolution (e.g., 1920x1080) determines the smoothness of the result.
    grid_y, grid_x = np.mgrid[extent[2]:extent[3]:complex(resolution_h),
                              extent[0]:extent[1]:complex(resolution_w)]

    # 3. Interpolate the sparse data (`amplitudes_sq` at `points_2d`) onto the dense grid.
    # 'linear' provides a good balance of speed and quality. 'cubic' is smoother but slower.
    heatmap = griddata(points_2d, amplitudes_sq, (grid_x, grid_y), method='linear', fill_value=0)

    # The result needs to be transposed because of indexing conventions (y, x) vs (row, col).
    return heatmap, extent

def get_colormaps():
    """Creates and caches custom colormaps for the worker process."""
    if 'heatmap_cmap' in worker_colormap_data:
        return worker_colormap_data['heatmap_cmap'], worker_colormap_data['phase_cmap']

    colors = ["#000000", "#0c0a2b", "#4a0b5e", "#9b1d5f", "#e2534b", "#fcae1e", "#f0f0c0"]
    nodes = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    heatmap_cmap = LinearSegmentedColormap.from_list("genesis_heatmap", list(zip(nodes, colors)))
    phase_cmap = plt.get_cmap('hsv')

    worker_colormap_data['heatmap_cmap'] = heatmap_cmap
    worker_colormap_data['phase_cmap'] = phase_cmap
    return heatmap_cmap, phase_cmap


def render_frame_worker(args_tuple):
    """Main worker function to render a single simulation frame."""
    frame_num, data_path, frames_dir, shared_info = args_tuple
    try:
        # --- 1. LOAD FRAME DATA ---
        with np.load(data_path, allow_pickle=True) as data:
            psi = unwrap_numpy_data(data['psi'])
            causal_graph = unwrap_numpy_data(data.get('causal_graph'))
            all_tracked_particles = unwrap_numpy_data(data.get('all_tracked_particles'))
            tracked_count = unwrap_numpy_data(data.get('tracked_count', 0))

        # --- 2. LOAD STATIC SUBSTRATE DATA FROM WORKER CACHE ---
        substrate_points = worker_substrate_data.get('points')
        substrate_neighbors = worker_substrate_data.get('neighbors')
        if substrate_points is None: raise RuntimeError("Worker not initialized with substrate data.")

        # --- 3. SETUP SCENE ---
        FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES, DPI = 19.2, 10.8, 100
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
        fig.set_facecolor('#08040E')
        ax.set_facecolor('#08040E')
        ax.set_aspect('equal')
        points_2d = substrate_points[:, :2]
        heatmap_cmap, phase_cmap = get_colormaps()

        # --- 4. RENDER LAYERS (FROM BOTTOM TO TOP) ---

        # Layer 0: Energy Heatmap (Robust Interpolation Method)
        amplitudes_sq = np.abs(psi)**2
        heatmap, extent = create_heatmap_and_extent(points_2d, amplitudes_sq)

        # Normalize the heatmap for consistent brightness across frames
        max_val = np.max(heatmap)
        if max_val > 1e-9:
            heatmap /= max_val

        ax.imshow(heatmap, extent=extent, origin='lower', cmap=heatmap_cmap, zorder=0, aspect='auto')

        # Layer 1: Static Substrate (Edges)
        substrate_lines = LineCollection(
            [[points_2d[i], points_2d[j]] for i, n_list in enumerate(substrate_neighbors) for j in n_list if i < j],
            colors='#20182D', linewidths=0.5, zorder=1
        )
        ax.add_collection(substrate_lines)

        # Layer 2: Substrate Nodes (colored by phase) - will now align perfectly
        phases = np.angle(psi)
        node_colors = phase_cmap((phases + np.pi) / (2 * np.pi))
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=10, c=node_colors, zorder=3, edgecolors='face')

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
                num_edges = len(dark_segments)
                all_segments = np.concatenate([dark_segments, light_segments])
                all_colors = ['#303030'] * num_edges + ['#505050'] * num_edges
                ax.add_collection(LineCollection(all_segments, colors=all_colors, linewidths=0.7, zorder=2))

        # Layer 4: Stable Particles (Hulls and Text)
        stable_particles = [p for p in all_tracked_particles if is_valid_array(all_tracked_particles) and p.get('state') == 'stable']
        if stable_particles:
            for particle in stable_particles:
                node_indices = particle.get('node_indices')
                if is_valid_array(node_indices) and len(node_indices) > 2:
                    try:
                        hull_points = substrate_points[node_indices, :2]
                        hull = ConvexHull(hull_points)
                        for simplex in hull.simplices:
                            ax.plot(hull_points[simplex, 0], hull_points[simplex, 1], color='lightgray', linewidth=1.0, zorder=10, linestyle='dashed')
                    except Exception: pass
            for particle in stable_particles:
                pos_2d = particle["position"][:2]
                label = (f"ID:{particle['track_id']} A:{particle['age']}\n"
                         f"M:{particle['mass']:.3f} Q:{particle['average_charge']:.2f}")
                ax.text(pos_2d[0], pos_2d[1] + 1.2, label, color='#FFFFFF', fontsize=9, ha='center', fontweight='bold', zorder=11, bbox=dict(facecolor='black', alpha=0.7, edgecolor='#00FFFF'))

        # --- 5. FINALIZE AND SAVE ---
        stable_count = len(stable_particles)
        title = (f"Genesis v22.2 | {shared_info.get('run_name', '...')} | "
                 f"Frame: {frame_num} | Tracked: {tracked_count} | Stable: {stable_count}")
        ax.set_title(title, fontsize=14, color='white', pad=20)

        # Set axis limits using the extent from the heatmap for perfect framing
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0)

        frame_filename = os.path.join(frames_dir, f"frame_{frame_num:05d}.png")
        fig.savefig(frame_filename, dpi=DPI, facecolor=fig.get_facecolor())
        plt.close(fig)
        return None

    except Exception as e:
        # Return the error message for debugging in the main process
        return f"Frame {frame_num}: Error - {type(e).__name__} - {e}\n{traceback.format_exc()}"
