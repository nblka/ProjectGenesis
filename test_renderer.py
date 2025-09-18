# test_renderer.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "Strategy-Aware Test"
# - The test script is updated to work with the new data-driven,
#   strategy-based rendering architecture.
# - It correctly reads `field_type` and `global_interaction_source_max` from
#   the metadata.json file.
# - It properly calls the worker initializer (`init_worker`) with the
#   correct arguments.
# - The arguments tuple passed to `render_frame_worker` is updated to match
#   the new, simplified signature.

import numpy as np
import argparse
import os
import json
import shutil

# --- Import centralized styling for console output ---
from styling import C, cprint

# --- Import the components to be tested ---
from renderer_worker import render_frame_worker, init_worker, worker_substrate_data

def main():
    """Main function to test the rendering of a single frame."""
    # --- 1. SETUP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(
        description="Render a single frame from a Project Genesis v16+ run for testing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('frame_number', type=int, help="The specific frame number to render.")
    parser.add_argument('-o', '--output', type=str, default=None, help="Optional output filename.")
    args = parser.parse_args()

    # --- 2. DEFINE PATHS AND VALIDATE ---
    RUN_DIR = args.run_directory
    FRAME_NUM = args.frame_number
    DATA_DIR = os.path.join(RUN_DIR, 'data')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')
    DATA_PATH = os.path.join(DATA_DIR, f"frame_{FRAME_NUM:05d}.npz")
    SUBSTRATE_PATH = os.path.join(RUN_DIR, 'substrate.npz')

    if not all(os.path.exists(p) for p in [METADATA_PATH, SUBSTRATE_PATH, DATA_PATH]):
        cprint(f"Error: Run directory '{RUN_DIR}' is incomplete.", C.ERROR); exit()

    cprint(f"--- Running Single Frame Renderer Test (v16.0) on Frame {FRAME_NUM} ---", C.SUBHEADER, attrs=C.BOLD_ATTR)

    # --- 3. LOAD METADATA AND SUBSTRATE ( mimicking the main script ) ---
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        with np.load(SUBSTRATE_PATH, allow_pickle=True) as sub_data:
            substrate_points = sub_data['points']
            substrate_neighbors = [list(n) for n in sub_data['neighbors']]

        global_max_source = metadata.get('global_interaction_source_max')
        if global_max_source is None:
            cprint("Warning: 'global_interaction_source_max' not in metadata. Using fallback.", C.WARNING)
            global_max_source = 1.0

    except Exception as e:
        cprint(f"Error loading necessary files: {e}", C.ERROR); exit()

    # --- 4. INITIALIZE THE 'WORKER' ENVIRONMENT IN THE MAIN THREAD ---
    cprint("Initializing worker environment...", C.WARNING)
    init_worker(substrate_points, substrate_neighbors, global_max_source)
    assert 'points' in worker_substrate_data, "Worker initialization failed."
    cprint("  > Worker initialized successfully.", C.SUCCESS)

    # --- 5. PREPARE ARGUMENTS AND CALL THE WORKER FUNCTION ---

    # Create a temporary directory for the output frame
    temp_output_dir = "temp_render_test"
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)

    # This is the tuple that the multiprocessing pool would normally create
    args_tuple = (FRAME_NUM, DATA_PATH, temp_output_dir, metadata)

    cprint(f"Calling render_frame_worker for frame {FRAME_NUM}...", C.WARNING)
    result = render_frame_worker(args_tuple)

    # --- 6. CHECK RESULTS AND CLEAN UP ---
    temp_output_file = os.path.join(temp_output_dir, f"frame_{FRAME_NUM:05d}.png")

    if result is None and os.path.exists(temp_output_file):
        # Determine final output name
        output_filename = args.output if args.output else f"test_render_frame_{FRAME_NUM}.png"
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Move the successful render from the temp dir to the final location
        shutil.move(temp_output_file, output_filename)
        cprint(f"\nSUCCESS! Test frame saved as '{os.path.abspath(output_filename)}'.", C.SUCCESS)
    else:
        cprint(f"\nFAILURE! Renderer returned an error:", C.ERROR)
        print(result)

    # Clean up the temporary directory
    shutil.rmtree(temp_output_dir)

    cprint("\n--- Test Complete ---", C.SUBHEADER, attrs=C.BOLD_ATTR)

if __name__ == "__main__":
    main()
