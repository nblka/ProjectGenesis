# test_renderer.py v14.3
# FIX: Imports the `init_worker` function from the correct module (`render_frames.py`)
#      instead of trying to find it in the worker module.

import numpy as np
import argparse
import os
import json
from termcolor import cprint

from renderer_worker import render_frame_worker, worker_substrate_data
from render_frames import init_worker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a single frame from a Project Genesis run for testing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('frame_number', type=int, help="The specific frame number to render.")
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()

    RUN_DIR = args.run_directory
    FRAME_NUM = args.frame_number

    DATA_DIR = os.path.join(RUN_DIR, 'data')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')
    DATA_PATH = os.path.join(DATA_DIR, f"frame_{FRAME_NUM:05d}.npz")
    SUBSTRATE_PATH = os.path.join(RUN_DIR, 'substrate.npz')

    if not os.path.exists(SUBSTRATE_PATH):
        cprint(f"Error: Substrate file not found at '{SUBSTRATE_PATH}'.", 'red'); exit()

    if not os.path.exists(METADATA_PATH) or not os.path.exists(DATA_PATH):
        cprint(f"Error: Required file(s) not found in '{RUN_DIR}'.", 'red'); exit()

    cprint(f"--- Running Single Frame Renderer on Frame {FRAME_NUM} ---", "cyan")

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    cprint("Initializing worker environment...", "yellow")
    with np.load(SUBSTRATE_PATH, allow_pickle=True) as sub_data:
        substrate_points = sub_data['points']
        substrate_neighbors = sub_data['neighbors']

    # CRITICAL STEP: Get the global max amplitude from the metadata
    global_max_amp_sq = metadata.get('global_max_amp_sq')
    if global_max_amp_sq is None or global_max_amp_sq < 1e-12:
        cprint(f"Warning: 'global_max_amp_sq' not found or is zero in metadata. Visualization might be inconsistent.", 'yellow')
        global_max_amp_sq = 1.0 # Fallback to prevent division by zero

    # Add this crucial piece of information to the shared metadata for workers
    metadata['global_max_amp_sq_for_norm'] = global_max_amp_sq

    init_worker(substrate_points, substrate_neighbors, global_max_amp_sq)

    assert 'points' in worker_substrate_data, "Worker initialization failed."

    output_dir = "."
    if args.output:
        output_filename_base = args.output
    else:
        output_filename_base = f"test_{metadata.get('run_name', 'run')}_frame_{FRAME_NUM}.png"

    temp_output_filename = os.path.join(output_dir, f"frame_{FRAME_NUM:05d}.png")
    if os.path.exists(output_filename_base):
        os.remove(output_filename_base)

    args_tuple = (FRAME_NUM, DATA_PATH, output_dir, metadata)

    cprint(f"Calling render_frame_worker for frame {FRAME_NUM}...", "yellow")
    result = render_frame_worker(args_tuple)

    if result is None and os.path.exists(temp_output_filename):
        os.rename(temp_output_filename, output_filename_base)
        cprint(f"SUCCESS! Test frame saved as '{os.path.abspath(output_filename_base)}'.", "green")
    else:
        cprint(f"FAILURE! Renderer returned an error: {result}", "red")
        if os.path.exists(temp_output_filename):
            os.remove(temp_output_filename)

    cprint("--- Test Complete ---", "cyan")
