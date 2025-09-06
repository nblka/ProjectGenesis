# render_video.py v6.1
"""
Project Genesis v6.1 - Visualization Core (Parallel & Robust)
----------------------------------------------------------------
- FEATURE: Frame rendering is now fully parallelized using multiprocessing,
  dramatically speeding up the process on multi-core CPUs.
- RETAINED: Uses the robust custom renderer (`PatchCollection`) to avoid
  matplotlib bugs.
- RETAINED: All robust data loading and error handling mechanisms.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend, crucial for multiprocessing
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import argparse
import os
from tqdm import tqdm
import glob
import subprocess
import shutil
import multiprocessing as mp
from termcolor import cprint

# === WORKER FUNCTION FOR PARALLEL RENDERING (MUST BE TOP-LEVEL) ===
def render_frame_worker(args_tuple):
    """
    Renders a single frame. This function is designed to be called by a multiprocessing pool.
    It takes all necessary data as an argument to be self-contained.
    """
    frame_idx, data_path, frames_dir, shared_info = args_tuple

    try:
        data = np.load(data_path, allow_pickle=True)

        # Robustly unwrap numpy object arrays
        def unwrap(d):
            if hasattr(d, 'ndim') and d.ndim == 0 and hasattr(d, 'dtype') and d.dtype == 'object':
                return d.item()
            return d

        points = unwrap(data['points'])
        psi = unwrap(data['psi'])
        simplices = unwrap(data['simplices'])
        stable_particles = unwrap(data['stable_particles'])
        tracked_count = unwrap(data['tracked_count'])

        # Create a figure and axes for this specific process
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('black')

        # --- Drawing Logic ---
        num_points = len(points)
        sim_size = 40 # Assuming fixed size

        # 1. Custom Background Renderer
        bg_colors_per_node = np.abs(psi)**2
        patches, face_colors = [], []
        cmap = plt.get_cmap('magma')
        norm = plt.Normalize(vmin=bg_colors_per_node.min(), vmax=bg_colors_per_node.max())

        if simplices.size > 0 and simplices.ndim == 2:
            for simplex in simplices:
                if np.any(simplex >= num_points): continue
                patches.append(Polygon(points[simplex], closed=True))
                avg_color_value = np.mean(bg_colors_per_node[simplex])
                face_colors.append(cmap(norm(avg_color_value)))

        p = PatchCollection(patches, facecolors=face_colors, edgecolors='gray', linewidth=0.3, alpha=0.8)
        ax.add_collection(p)

        # 2. Nodes (color=phase, size=amplitude)
        phases, amplitudes = np.angle(psi), np.abs(psi)
        node_sizes = (amplitudes * 80)**1.5 + 5
        ax.scatter(points[:, 0], points[:, 1], c=phases, cmap='hsv', s=node_sizes, zorder=2, edgecolors='black', linewidth=0.2)

        # 3. Particle Info
        if isinstance(stable_particles, (list, np.ndarray)) and len(stable_particles) > 0:
            for p_item in stable_particles:
                pos, track_id = p_item["position"], p_item["track_id"]
                label = f"ID:{track_id}\nM={p_item['mass']:.1f}, Q={p_item['average_charge']:.2f}\nAge:{p_item['age']}"
                x_offset, y_offset, ha, va = (1, 1, 'left', 'bottom')
                if pos[0] > sim_size * 0.75: x_offset, ha = -1, 'right'
                if pos[1] > sim_size * 0.75: y_offset, va = -1, 'top'
                box_pos = (pos[0] + x_offset, pos[1] + y_offset)
                ax.text(box_pos[0], box_pos[1], label, color='lime', fontsize=8, ha=ha, va=va,
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='lime', boxstyle='round,pad=0.3'))
                ax.plot([pos[0], box_pos[0]], [pos[1], box_pos[1]], ':', color='lime', alpha=0.7, lw=1)
                highlight_nodes = np.append(p_item.get('neighbors', []), p_item['id'])
                for node_idx in highlight_nodes:
                    if node_idx < len(points):
                        node_pos = points[node_idx]
                        circle = plt.Circle(node_pos, 0.5, color='white', fill=False, alpha=0.8, lw=1.5)
                        ax.add_artist(circle)

        # 4. Title and Layout
        ax.set_title(
            f"Project Genesis | SEED: {shared_info['seed']} | Mode: {shared_info['mode']} | Nodes: {num_points} | Frame: {frame_idx+1}\n"
            f"Tracked: {tracked_count} | Confirmed Stable: {len(stable_particles) if isinstance(stable_particles, (list, np.ndarray)) else 0} (Goal: {shared_info['goal']})",
            fontsize=10
        )
        ax.set_aspect('equal'); ax.set_xlim(0, sim_size); ax.set_ylim(0, sim_size)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()

        # Save the figure and close it
        frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        fig.savefig(frame_filename, dpi=150)
        plt.close(fig)
        return None # Success
    except Exception as e:
        return f"Frame {frame_idx+1}: {type(e).__name__} - {e}" # Failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a video from Project Genesis simulation data.")
    parser.add_argument('seed', type=int, help="The SEED of the simulation data to render.")
    parser.add_argument('--keep-frames', action='store_true', help="Keep individual frame PNGs after rendering.")
    # Add args to get info for the title
    parser.add_argument('--mode', type=str, default='soup', help="Simulation mode used.")
    parser.add_argument('--points', type=int, default=2000, help="Number of nodes used.")
    parser.add_argument('--goal', type=int, default=10, help="Stable particle goal used.")
    args = parser.parse_args()

    SEED = args.seed
    DATA_DIR = f"data_SEED_{SEED}"
    FRAMES_DIR = f"frames_SEED_{SEED}"
    OUTPUT_FILENAME = f'genesis_sim_SEED_{SEED}_v6.1.mp4'

    if not os.path.exists(DATA_DIR): cprint(f"Error: Data directory '{DATA_DIR}' not found.", 'red'); exit()
    if not os.path.exists(FRAMES_DIR): os.makedirs(FRAMES_DIR)

    data_files = sorted(glob.glob(os.path.join(DATA_DIR, 'frame_*.npz')))
    if not data_files: cprint(f"Error: No frame data found in '{DATA_DIR}'.", 'red'); exit()
    print(f"Found {len(data_files)} data frames to render.")

    # --- MAIN PARALLEL RENDERING LOOP ---
    print("Stage 1: Rendering individual frames in parallel...")

    shared_info = {'seed': SEED, 'mode': args.mode, 'points': args.points, 'goal': args.goal}
    tasks = [(i, path, FRAMES_DIR, shared_info) for i, path in enumerate(data_files)]
    failed_frames = []

    # Use all available CPU cores
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # imap_unordered is great for this as frame order doesn't matter yet
        results = list(tqdm(pool.imap_unordered(render_frame_worker, tasks), total=len(tasks), desc="Rendering PNGs"))

    for res in results:
        if res is not None:
            failed_frames.append(res)

    if failed_frames:
        cprint(f"\nWarning: {len(failed_frames)} frames failed to render. See details below:", 'yellow')
        for fail_log in failed_frames[:5]: # Print first 5 errors
            cprint(f" - {fail_log}", 'red')
        if len(failed_frames) > 5:
            cprint(f"   ... and {len(failed_frames) - 5} more.", 'red')

    # --- VIDEO COMPILATION ---
    print("\nStage 2: Compiling frames into video...")
    ffmpeg_command = [
        'ffmpeg', '-framerate', '20', '-i', f'{FRAMES_DIR}/frame_%05d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', OUTPUT_FILENAME
    ]
    try:
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        print(f"Video compilation successful! Output: '{OUTPUT_FILENAME}'")
    except subprocess.CalledProcessError as e:
        cprint("\n--- FFMPEG ERROR ---", 'red')
        cprint(e.stderr, 'red')
        cprint("Video compilation failed. The PNG frames are still in the frames directory.", 'yellow')
    except FileNotFoundError:
        cprint("\nError: `ffmpeg` command not found. Please ensure it's installed and in your system's PATH.", 'red')

    # --- CLEANUP ---
    if not args.keep_frames and os.path.exists(FRAMES_DIR):
        print(f"Cleaning up temporary frames directory: '{FRAMES_DIR}'")
        shutil.rmtree(FRAMES_DIR)

    print("\nProcess complete.")
