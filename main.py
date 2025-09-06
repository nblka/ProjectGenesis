# run_and_render.py v6.2
"""
Project Genesis v6.2 - Unified Workflow
---------------------------------------
- This script combines simulation and rendering into a single, robust workflow.
- It first runs the simulation to generate raw data files.
- Then, it automatically starts the parallelized rendering process.
- This is the final, definitive version that incorporates all bug fixes and features.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import argparse
import os
from tqdm import tqdm
import glob
import subprocess
import shutil
import time
import multiprocessing as mp
from termcolor import cprint

# Import our modular components
from simulation import WavePacketUniverse, PrimordialSoupUniverse
from tracker import ParticleTracker

# === WORKER FUNCTION FOR PARALLEL RENDERING (MUST BE TOP-LEVEL) ===
def render_frame_worker(args_tuple):
    frame_idx, data_path, frames_dir, shared_info = args_tuple
    try:
        def unwrap(d):
            if hasattr(d, 'ndim') and d.ndim == 0 and hasattr(d, 'dtype') and d.dtype == 'object':
                return d.item()
            return d

        data = np.load(data_path, allow_pickle=True)
        points = unwrap(data['points']); psi = unwrap(data['psi']); simplices = unwrap(data['simplices'])
        stable_particles = unwrap(data['stable_particles']); tracked_count = unwrap(data['tracked_count'])

        fig, ax = plt.subplots(figsize=(10, 10)); ax.set_facecolor('black')
        num_points, sim_size = len(points), 40

        bg_colors_per_node = np.abs(psi)**2
        patches, face_colors = [], []
        cmap, norm = plt.get_cmap('magma'), plt.Normalize(vmin=bg_colors_per_node.min(), vmax=bg_colors_per_node.max())
        if simplices.size > 0 and simplices.ndim == 2:
            for simplex in simplices:
                if np.any(simplex >= num_points): continue
                patches.append(Polygon(points[simplex], closed=True))
                face_colors.append(cmap(norm(np.mean(bg_colors_per_node[simplex]))))
        ax.add_collection(PatchCollection(patches, facecolors=face_colors, edgecolors='gray', linewidth=0.3, alpha=0.8))

        phases, amplitudes = np.angle(psi), np.abs(psi)
        node_sizes = (amplitudes * 80)**1.5 + 5
        ax.scatter(points[:, 0], points[:, 1], c=phases, cmap='hsv', s=node_sizes, zorder=2, edgecolors='black', linewidth=0.2)

        if isinstance(stable_particles, (list, np.ndarray)) and len(stable_particles) > 0:
            for p_item in stable_particles:
                pos, track_id = p_item["position"], p_item["track_id"]
                label = f"ID:{track_id}\nM={p_item['mass']:.1f}, Q={p_item['average_charge']:.2f}\nAge:{p_item['age']}"
                x_offset, y_offset, ha, va = (1, 1, 'left', 'bottom')
                if pos[0] > sim_size * 0.75: x_offset, ha = -1, 'right'
                if pos[1] > sim_size * 0.75: y_offset, va = -1, 'top'
                box_pos = (pos[0] + x_offset, pos[1] + y_offset)
                ax.text(box_pos[0], box_pos[1], label, color='lime', fontsize=8, ha=ha, va=va, bbox=dict(facecolor='black', alpha=0.7, edgecolor='lime', boxstyle='round,pad=0.3'))
                ax.plot([pos[0], box_pos[0]], [pos[1], box_pos[1]], ':', color='lime', alpha=0.7, lw=1)
                highlight_nodes = np.append(p_item.get('neighbors', []), p_item['id'])
                for node_idx in highlight_nodes:
                    if node_idx < len(points):
                        node_pos = points[node_idx]
                        circle = plt.Circle(node_pos, 0.5, color='white', fill=False, alpha=0.8, lw=1.5)
                        ax.add_artist(circle)

        ax.set_title(
            f"Project Genesis | SEED: {shared_info['seed']} | Mode: {shared_info['mode']} | Nodes: {num_points} | Frame: {frame_idx+1}\n"
            f"Tracked: {tracked_count} | Confirmed Stable: {len(stable_particles) if isinstance(stable_particles, (list, np.ndarray)) else 0} (Goal: {shared_info['goal']})",
            fontsize=10)
        ax.set_aspect('equal'); ax.set_xlim(0, sim_size); ax.set_ylim(0, sim_size)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()

        frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        fig.savefig(frame_filename, dpi=150)
        plt.close(fig)
        return None
    except Exception as e:
        return f"Frame {frame_idx+1}: {type(e).__name__} - {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Project Genesis simulation and render the video.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, help="Specify a seed for reproducibility. If not given, a random one is used.")
    parser.add_argument('--points', type=int, default=2000, help="Number of points.")
    parser.add_argument('--mode', type=str, default='soup', choices=['packet', 'soup'], help=("'packet': Start with a single, coherent wave packet.\n" "'soup': Start from a high-energy, random primordial soup."))
    parser.add_argument('--frames', type=int, default=1200, help="Maximum number of frames.")
    parser.add_argument('--stable_particles', type=int, default=10, help="Target number of stable particles to find.")
    parser.add_argument('--keep-frames', action='store_true', help="Keep the individual frame PNG files after rendering the video.")
    args = parser.parse_args()

    SEED = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
    np.random.seed(SEED)
    USE_MULTIPROCESSING = mp.cpu_count() > 1

    DATA_DIR = f"data_SEED_{SEED}"
    FRAMES_DIR = f"frames_SEED_{SEED}"
    OUTPUT_FILENAME = f'genesis_sim_SEED_{SEED}_v6.2.mp4'

    # --- STAGE 1: SIMULATION ---
    print("\n--- Project Genesis v6.2 ---")
    print(f"Using SEED: {SEED}")
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    print(f"Raw data will be saved in: '{DATA_DIR}'")

    sim = WavePacketUniverse(num_points=args.points, use_multiprocessing=USE_MULTIPROCESSING) if args.mode == 'packet' else PrimordialSoupUniverse(num_points=args.points, use_multiprocessing=USE_MULTIPROCESSING, initial_energy=5.0)
    tracker = ParticleTracker(pool=sim.pool)
    print("4. Launching simulation...")
    start_time = time.time()

    for frame in tqdm(range(args.frames), desc="Simulating Frames"):
        sim.update_step()
        stable_particles = tracker.analyze_frame(sim, frame + 1)
        frame_filename = os.path.join(DATA_DIR, f"frame_{frame:05d}.npz")
        np.savez_compressed(frame_filename, points=sim.points, psi=sim.psi, simplices=sim.simplices, stable_particles=np.array(stable_particles, dtype=object), tracked_count=len(tracker.tracked_particles))
        if len(stable_particles) >= args.stable_particles:
            print(f"\nSUCCESS: Goal of {args.stable_particles} stable particle(s) reached at frame {frame + 1}.")
            break

    sim.close_pool()
    print(f"Simulation complete. Total time: {time.time() - start_time:.2f} seconds.")

    # --- STAGE 2: RENDERING ---
    if not os.path.exists(FRAMES_DIR): os.makedirs(FRAMES_DIR)
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, 'frame_*.npz')))
    if not data_files:
        cprint(f"Error: No frame data found in '{DATA_DIR}'.", 'red'); exit()
    print(f"\nFound {len(data_files)} data frames to render.")

    print("Stage 2.1: Rendering individual frames in parallel...")
    shared_info = {'seed': SEED, 'mode': args.mode, 'points': args.points, 'goal': args.stable_particles}
    tasks = [(i, path, FRAMES_DIR, shared_info) for i, path in enumerate(data_files)]
    failed_frames = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(render_frame_worker, tasks), total=len(tasks), desc="Rendering PNGs"))
    for res in results:
        if res is not None: failed_frames.append(res)

    # ... (Video compilation and cleanup logic is unchanged)
    print("\nStage 2.2: Compiling frames into video...")
    ffmpeg_command = ['ffmpeg', '-framerate', '20', '-i', f'{FRAMES_DIR}/frame_%05d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', OUTPUT_FILENAME]
    try:
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        print(f"Video compilation successful! Output: '{OUTPUT_FILENAME}'")
    except subprocess.CalledProcessError as e:
        cprint("\n--- FFMPEG ERROR ---", 'red'); print(e.stderr); print("--------------------")
    except FileNotFoundError:
        cprint("\nError: `ffmpeg` command not found.", 'red')

    if not args.keep_frames:
        print(f"Cleaning up temporary data and frames directories...")
        shutil.rmtree(DATA_DIR); shutil.rmtree(FRAMES_DIR)
    print("\nProcess complete.")
