# render_frames.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "Strategy-Aware Orchestrator"
# - The main orchestrator now reads the `field_type` and the
#   `global_interaction_source_max` from the `metadata.json` file.
# - It passes the `field_type` to the worker processes, allowing them
#   to select the correct visualization strategy.
# - The `global_interaction_source_max` is used to initialize the worker's
#   logarithmic normalization, ensuring consistent heatmap brightness.

import numpy as np
import argparse
import os
import glob
import shutil
import json
import multiprocessing as mp
from tqdm import tqdm

# --- Import centralized styling for console output ---
from styling import C, cprint

# --- Import the worker function and its initializer ---
from renderer_worker import render_frame_worker, init_worker

def get_frame_number(path):
    """Extracts the integer frame number from a file path for correct sorting."""
    try:
        return int(os.path.basename(path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        return -1

def main():
    """Main function to orchestrate the frame rendering process."""
    # Set the multiprocessing start method for cross-platform compatibility
    if os.name != 'posix':
        mp.set_start_method('spawn', force=True)

    # --- 1. SETUP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Render frames from a Project Genesis v16+ run.")
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('-rs', '--render-step', type=int, default=1, help="Render every N-th frame to speed up preview generation.")
    args = parser.parse_args()

    # --- 2. DEFINE PATHS AND VALIDATE RUN DIRECTORY ---
    RUN_DIR = args.run_directory
    DATA_DIR = os.path.join(RUN_DIR, 'data')
    FRAMES_DIR = os.path.join(RUN_DIR, 'frames')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')
    SUBSTRATE_PATH = os.path.join(RUN_DIR, 'substrate.npz')

    if not all(os.path.exists(p) for p in [METADATA_PATH, SUBSTRATE_PATH, DATA_DIR]):
        cprint(f"Error: Run directory '{RUN_DIR}' is incomplete. Missing metadata, substrate, or data.", C.ERROR); exit()

    # --- 3. LOAD METADATA AND HEAVY SUBSTRATE DATA (ONCE) ---
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        cprint(f"\n--- STAGE 2: RENDERING FRAMES for run '{metadata['run_name']}' ---", C.SUBHEADER, attrs=C.BOLD_ATTR)

        cprint("Loading static substrate data...", C.WARNING)
        with np.load(SUBSTRATE_PATH, allow_pickle=True) as sub_data:
            substrate_points_np = sub_data['points']
            substrate_neighbors_list = [list(n) for n in sub_data['neighbors']]

        # CRITICAL STEP: Get the global max source value from metadata for normalization.
        global_max_source = metadata.get('global_interaction_source_max')
        if global_max_source is None or global_max_source < 1e-12:
            cprint(f"Warning: 'global_interaction_source_max' not found in metadata. Visualization might be inconsistent.", C.WARNING)
            global_max_source = 1.0 # Fallback

    except Exception as e:
        cprint(f"Error loading metadata or substrate files: {e}", C.ERROR); exit()

    # --- 4. PREPARE FRAME DIRECTORY AND TASK LIST ---
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR)

    all_data_files = sorted(glob.glob(os.path.join(DATA_DIR, 'frame_*.npz')), key=get_frame_number)
    if not all_data_files:
        cprint(f"Error: No data files (.npz) found in '{DATA_DIR}'.", C.ERROR); exit()

    data_files_to_render = [p for p in all_data_files if get_frame_number(p) % args.render_step == 0]

    # The 'shared_info' dictionary will be passed to every worker.
    # It contains all the run-specific info they need.
    tasks = [(get_frame_number(path), path, FRAMES_DIR, metadata) for path in data_files_to_render]

    cprint(f"Found {len(tasks)} frames to render.", C.WARNING)

    # --- 5. EXECUTE RENDERING USING MULTIPROCESSING ---
    if tasks:
        # The arguments for the initializer are the heavy numpy arrays that each worker needs.
        init_args = (substrate_points_np, substrate_neighbors_list, global_max_source)

        num_processes = min(mp.cpu_count(), len(tasks))
        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=init_args) as pool:
            results = list(tqdm(pool.imap_unordered(render_frame_worker, tasks), total=len(tasks), desc="Rendering Frames", bar_format="{l_bar}{bar:30}{r_bar}"))

        failed_frames = [res for res in results if res is not None]
        if failed_frames:
            cprint(f"\nWarning: {len(failed_frames)} frame(s) failed to render.", C.WARNING)

    cprint("Rendering complete.", C.SUCCESS)
    cprint(f"\nFrames saved in '{FRAMES_DIR}'.", C.SUCCESS)
    cprint(f"To compile the video, run:\n"
           f"  python compile_video.py {RUN_DIR}", C.SUCCESS)

if __name__ == "__main__":
    main()
