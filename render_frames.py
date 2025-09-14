# render_frames.py v14.3
# Part of Project Genesis: Breathing Causality
# v14.3: CRITICAL FIX - Aligns the data loading mechanism with the new
#        data storage architecture.
#        - Reads lightweight metadata from `metadata.json`.
#        - Reads heavyweight, static substrate data (`points`, `neighbors`)
#          from the dedicated `substrate.npz` file.
#        - This makes the script robust and scalable.

import numpy as np
import argparse
import os
import glob
import shutil
import json
import multiprocessing as mp
from tqdm import tqdm
from termcolor import cprint

from renderer_worker import render_frame_worker, worker_substrate_data

def get_frame_number(path):
    """Extracts the frame number from a file path."""
    try:
        return int(os.path.basename(path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        return -1

def init_worker(points, neighbors):
    """Initializes each worker process with the large, static substrate data."""
    worker_substrate_data['points'] = points
    worker_substrate_data['neighbors'] = neighbors

if __name__ == "__main__":
    if os.name != 'posix':
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Render frames from a Project Genesis v14.3 run.")
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('-rs', '--render-step', type=int, default=1, help="Render every N-th frame.")
    args = parser.parse_args()

    RUN_DIR = args.run_directory
    DATA_DIR = os.path.join(RUN_DIR, 'data')
    FRAMES_DIR = os.path.join(RUN_DIR, 'frames')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')
    # --- NEW: Define path for the substrate data file ---
    SUBSTRATE_PATH = os.path.join(RUN_DIR, 'substrate.npz')

    # --- Robust File Checks ---
    if not os.path.exists(METADATA_PATH):
        cprint(f"Error: metadata.json not found in '{RUN_DIR}'.", 'red'); exit()
    if not os.path.exists(SUBSTRATE_PATH):
        cprint(f"Error: substrate.npz not found in '{RUN_DIR}'. Run the simulation first.", 'red'); exit()

    # --- Load Metadata and Substrate Data ---
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        cprint(f"\n--- STAGE 2: RENDERING FRAMES for run '{metadata['run_name']}' ---", 'cyan')

        cprint("Loading static substrate from substrate.npz...", 'yellow')
        with np.load(SUBSTRATE_PATH, allow_pickle=True) as sub_data:
            substrate_points_np = sub_data['points']
            substrate_neighbors_list = sub_data['neighbors']

    except Exception as e:
        cprint(f"Error loading metadata or substrate files: {e}", 'red'); exit()

    # --- Directory and File Setup ---
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR)

    all_data_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npz')), key=get_frame_number)
    if not all_data_files:
        cprint(f"Error: No data files (.npz) found in '{DATA_DIR}'.", 'red'); exit()

    data_files_to_render = [p for p in all_data_files if get_frame_number(p) % args.render_step == 0 and get_frame_number(p) != -1]

    # --- Prepare for Multiprocessing ---
    tasks = [(get_frame_number(path), path, FRAMES_DIR, metadata) for path in data_files_to_render]

    cprint(f"Found {len(tasks)} frames to render.", 'yellow')

    if tasks:
        # The arguments for the initializer are the heavy numpy arrays
        init_args = (substrate_points_np, substrate_neighbors_list)

        with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=init_args) as pool:
            results = list(tqdm(pool.imap_unordered(render_frame_worker, tasks), total=len(tasks), desc="Rendering Frames"))

        failed_frames = [res for res in results if res is not None]
        if failed_frames:
            cprint(f"\nWarning: {len(failed_frames)} frame(s) failed to render.", 'yellow')

    cprint("Rendering complete.", 'green')
    cprint(f"\nFrames saved in '{FRAMES_DIR}'.", 'green')
