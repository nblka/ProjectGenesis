# render_frames.py v14.4
# Part of Project Genesis: Breathing Causality
# v14.4: "Integrated Normalization" - A more robust and efficient version.
# - The pre-scan for the global maximum amplitude is REMOVED.
# - Instead, it now reads the `global_max_amp_sq` value directly from the
#   `metadata.json` file, which is calculated and saved by the simulation.
# - This is faster, more elegant, and ensures perfect visualization consistency.

import numpy as np
import argparse
import os
import glob
import shutil
import json
import multiprocessing as mp
from tqdm import tqdm
from termcolor import cprint

# Import the worker function and its initializer from the separate module
from renderer_worker import render_frame_worker, init_worker

def get_frame_number(path):
    """Extracts the integer frame number from a file path for correct sorting."""
    try:
        return int(os.path.basename(path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        return -1

if __name__ == "__main__":
    # Set the multiprocessing start method for cross-platform compatibility (macOS/Windows)
    if os.name != 'posix':
        mp.set_start_method('spawn', force=True)

    # --- 1. SETUP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Render frames from a Project Genesis v15+ run.")
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('-rs', '--render-step', type=int, default=1, help="Render every N-th frame to speed up preview generation.")
    args = parser.parse_args()

    # --- 2. DEFINE PATHS AND VALIDATE RUN DIRECTORY ---
    RUN_DIR = args.run_directory
    DATA_DIR = os.path.join(RUN_DIR, 'data')
    FRAMES_DIR = os.path.join(RUN_DIR, 'frames')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')
    SUBSTRATE_PATH = os.path.join(RUN_DIR, 'substrate.npz')

    if not os.path.exists(METADATA_PATH):
        cprint(f"Error: metadata.json not found in '{RUN_DIR}'.", 'red'); exit()
    if not os.path.exists(SUBSTRATE_PATH):
        cprint(f"Error: substrate.npz not found in '{RUN_DIR}'. Run the simulation first.", 'red'); exit()

    # --- 3. LOAD METADATA AND HEAVY SUBSTRATE DATA (ONCE) ---
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        cprint(f"\n--- STAGE 2: RENDERING FRAMES for run '{metadata['run_name']}' ---", 'cyan')

        cprint("Loading static substrate data from substrate.npz...", 'yellow')
        with np.load(SUBSTRATE_PATH, allow_pickle=True) as sub_data:
            substrate_points_np = sub_data['points']
            # Convert neighbor data to a list of lists for consistent processing
            substrate_neighbors_list = [list(n) for n in sub_data['neighbors']]

        # CRITICAL STEP: Get the global max amplitude from the metadata
        global_max_amp_sq = metadata.get('global_max_amp_sq')
        if global_max_amp_sq is None or global_max_amp_sq < 1e-12:
            cprint(f"Warning: 'global_max_amp_sq' not found or is zero in metadata. Visualization might be inconsistent.", 'yellow')
            global_max_amp_sq = 1.0 # Fallback to prevent division by zero

        # Add this crucial piece of information to the shared metadata for workers
        metadata['global_max_amp_sq_for_norm'] = global_max_amp_sq

    except Exception as e:
        cprint(f"Error loading metadata or substrate files: {e}", 'red'); exit()

    # --- 4. PREPARE FRAME DIRECTORY AND TASK LIST ---
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR)

    all_data_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npz')), key=get_frame_number)
    if not all_data_files:
        cprint(f"Error: No data files (.npz) found in '{DATA_DIR}'.", 'red'); exit()

    # Filter frames based on the render-step argument
    data_files_to_render = [p for p in all_data_files if get_frame_number(p) % args.render_step == 0]
    tasks = [(get_frame_number(path), path, FRAMES_DIR, metadata) for path in data_files_to_render]

    cprint(f"Found {len(tasks)} frames to render.", 'yellow')

    # --- 5. EXECUTE RENDERING USING MULTIPROCESSING ---
    if tasks:
        # The arguments for the initializer are the heavy numpy arrays that each worker needs
        init_args = (substrate_points_np, substrate_neighbors_list, global_max_amp_sq)

        with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=init_args) as pool:
            # Use imap_unordered for efficiency, as frame render order doesn't matter
            results = list(tqdm(pool.imap_unordered(render_frame_worker, tasks), total=len(tasks), desc="Rendering Frames"))

        # Check for any errors returned by the workers
        failed_frames = [res for res in results if res is not None]
        if failed_frames:
            cprint(f"\nWarning: {len(failed_frames)} frame(s) failed to render.", 'yellow')
            # for fail_log in failed_frames: print(fail_log) # Uncomment for detailed error logs

    cprint("Rendering complete.", 'green')
    cprint(f"\nFrames saved in '{FRAMES_DIR}'.", 'green')
