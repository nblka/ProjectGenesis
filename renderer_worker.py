# renderer_worker.py v16.1
# Part of Project Genesis: Breathing Causality
# v16.1: "Colormap Dependency Fix"
# - CRITICAL FIX: The worker now correctly manages colormaps and passes the
#   appropriate colormap object to the visualization strategy methods.
# - It no longer tries to access colormaps as attributes of the strategy object.
# - Imports `HEATMAP_CMAP` and `PHASE_CMAP` from the central `styling` module.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import os
import traceback

# --- Import centralized styling, INCLUDING COLOMAPS ---
from styling import (
    COLOR_BACKGROUND, COLOR_SUBSTRATE_EDGES, COLOR_CAUSAL_DARK,
    COLOR_CAUSAL_LIGHT, COLOR_PARTICLE_HULL_FACE, COLOR_PARTICLE_HULL_EDGE,
    HEATMAP_CMAP, PHASE_CMAP # IMPORT THE COLOMAPS
)
# --- Import visualization strategies ---
from visualization_strategies import ScalarFieldViz

# --- Worker Caching (no changes) ---
worker_substrate_data = {}
worker_log_norm_data = {}

# (init_worker, unwrap_numpy_data, is_valid_array, create_heatmap_and_extent functions remain unchanged)
def init_worker(points, neighbors, global_interaction_source_max):
    """Initializes each worker process with large, static data."""
    worker_substrate_data['points'] = points
    worker_substrate_data['neighbors'] = neighbors
    epsilon = 1e-3
    log_max = np.log1p(global_interaction_source_max / epsilon) if global_interaction_source_max > 0 else 1.0
    worker_log_norm_data['epsilon'] = epsilon
    worker_log_norm_data['log_max'] = log_max

def unwrap_numpy_data(d):
    if hasattr(d, 'ndim') and d.ndim == 0: return d.item()
    return d

def is_valid_array(arr):
    if isinstance(arr, np.ndarray): return arr.size > 0
    if isinstance(arr, (list, tuple)): return len(arr) > 0
    return False

def create_heatmap_and_extent(points_2d, values, resolution_h, resolution_w):
    x_coords, y_coords = points_2d[:, 0], points_2d[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    padding_x = (x_max - x_min) * 0.02
    padding_y = (y_max - y_min) * 0.02
    extent = [x_min - padding_x, x_max + padding_x, y_min - padding_y, y_max + padding_y]
    grid_y, grid_x = np.mgrid[extent[2]:extent[3]:complex(resolution_h), extent[0]:extent[1]:complex(resolution_w)]
    heatmap = griddata(points_2d, values, (grid_x, grid_y), method='linear', fill_value=0)
    return heatmap, extent

def render_frame_worker(args_tuple):
    """Main worker function to render a single simulation frame."""
    frame_num, data_path, frames_dir, shared_info = args_tuple
    try:
        # ... (Loading data and cached data remains the same) ...
        with np.load(data_path, allow_pickle=True) as data:
            field_values = unwrap_numpy_data(data['field_values'])
            causal_graph = unwrap_numpy_data(data.get('causal_graph'))
            stable_attractors = unwrap_numpy_data(data.get('stable_attractors'))
        substrate_points = worker_substrate_data.get('points')
        substrate_neighbors = worker_substrate_data.get('neighbors')
        epsilon = worker_log_norm_data.get('epsilon')
        log_max = worker_log_norm_data.get('log_max')

        # --- Select Visualization Strategy (unchanged) ---
        field_type = shared_info.get('field_type', 'scalar')
        if field_type == 'scalar':
            viz_strategy = ScalarFieldViz()
        else:
            raise ValueError(f"Unknown field_type for visualization: {field_type}")

        # --- Setup Scene (unchanged) ---
        FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES, DPI = 19.2, 10.8, 100
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), dpi=DPI)
        fig.set_facecolor(COLOR_BACKGROUND)
        ax.set_facecolor(COLOR_BACKGROUND)
        ax.set_aspect('equal')
        points_2d = substrate_points[:, :2]

        # --- RENDER LAYERS (with corrected calls) ---

        # Layer 0: Energy Heatmap
        heatmap_values = viz_strategy.get_heatmap_values(field_values)
        log_values = np.log1p(heatmap_values / epsilon)
        heatmap, extent = create_heatmap_and_extent(points_2d, log_values, resolution_h=540, resolution_w=960)
        normalized_heatmap = heatmap / log_max if log_max > 1e-9 else heatmap
        # --- FIX: Use the imported HEATMAP_CMAP directly ---
        ax.imshow(normalized_heatmap, extent=extent, origin='lower', cmap=HEATMAP_CMAP, zorder=0, aspect='auto', vmin=0.0, vmax=log_max)

        # Layer 1: Static Substrate Edges (unchanged)
        ax.add_collection(LineCollection(
            [[points_2d[i], points_2d[j]] for i, n_list in enumerate(substrate_neighbors) for j in n_list if i < j],
            colors=COLOR_SUBSTRATE_EDGES, linewidths=0.5, zorder=1
        ))

        # Layer 2: Substrate Nodes
        # --- FIX: Pass the imported PHASE_CMAP to the strategy method ---
        node_colors = viz_strategy.get_node_colors(field_values, PHASE_CMAP)
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=8, c=node_colors, zorder=3, edgecolors='face', alpha=0.8)

        # ... (Layers 3 and 4 for Causal Graph and Particles remain unchanged) ...
        # Layer 3: Dynamic Causal Graph
        if is_valid_array(causal_graph):
            source_indices = np.concatenate([inc for inc in causal_graph if is_valid_array(inc)])
            repeat_counts = [len(inc) if is_valid_array(inc) else 0 for inc in causal_graph]
            target_indices = np.repeat(np.arange(len(causal_graph)), repeat_counts)
            if source_indices.size > 0:
                p_sources = points_2d[source_indices]
                p_targets = points_2d[target_indices]
                p_midpoints = (p_sources + p_targets) / 2.0
                dark_segments = np.array(list(zip(p_sources, p_midpoints)))
                light_segments = np.array(list(zip(p_midpoints, p_targets)))
                ax.add_collection(LineCollection(dark_segments, colors=COLOR_CAUSAL_DARK, linewidths=0.7, zorder=2))
                ax.add_collection(LineCollection(light_segments, colors=COLOR_CAUSAL_LIGHT, linewidths=0.7, zorder=2))

        # Layer 4: Stable Particles (Hulls and Text)
        if is_valid_array(stable_attractors):
            hulls = []
            for particle in stable_attractors:
                node_indices = particle.get('node_indices')
                if is_valid_array(node_indices) and len(node_indices) >= 3:
                    hull_points = points_2d[node_indices]
                    try:
                        hull = ConvexHull(hull_points)
                        polygon = Polygon(hull_points[hull.vertices], closed=True)
                        hulls.append(polygon)
                        pos_2d = particle["position"][:2]
                        label = (f"ID:{particle['track_id']} A:{particle['age']}\n"
                                f"M:{particle['mass']:.3f} Q:{particle['average_charge']:.2f}")
                        ax.text(pos_2d[0], pos_2d[1] + 1.2, label, color='#FFFFFF', fontsize=9, ha='center', alpha=0.7,
                                fontweight='bold', zorder=11, bbox=dict(facecolor='black', alpha=0.5, edgecolor='#FFD700'))
                    except Exception: pass
            if hulls:
                ax.add_collection(PatchCollection(hulls, facecolor=COLOR_PARTICLE_HULL_FACE, edgecolor=COLOR_PARTICLE_HULL_EDGE,
                                                  linewidth=1.5, linestyles='--', alpha=0.25, zorder=4))

        # --- Finalize and Save (unchanged) ---
        tracked_count = shared_info.get('tracked_particle_count_for_frame', {}).get(str(frame_num), 0)
        stable_count = len(stable_attractors) if is_valid_array(stable_attractors) else 0
        title = (f"Genesis v16.1 | {shared_info.get('run_name', '...')} | Field: {field_type.capitalize()} | "
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
