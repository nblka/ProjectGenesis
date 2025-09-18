# main.py v16.0
# Part of Project Genesis: Breathing Causality
# v16.0: "The Abstract Orchestrator"
# - Fully integrates the new abstract architecture (Field, IC, Causality).
# - The main function now follows a clear Dependency Injection pattern:
#   1. Create the substrate (TopologyData).
#   2. Create the appropriate Field object (e.g., ScalarField).
#   3. Use an Initial Condition generator to populate the Field object.
#   4. Select the Causality computer.
#   5. Inject all these components into the Simulation engine.
# - Metadata is updated to include the field type for the renderer's benefit.

import numpy as np
import argparse
import os
import shutil
import time
import json
from tqdm import tqdm

# --- Import centralized styling for console output ---
from styling import C, cprint

# --- Import all components from the new architecture ---
from topologies import TopologyFactory
from field import ScalarField # For now, we only have ScalarField
from initial_conditions import PrimordialSoupState, WavePacketState, VortexState
from causality import ConvergentCausality, DivergentCausality
from simulation import Simulation
from tracker import ParticleTracker
from analytics import GlobalAnalytics

def main():
    """Main function to run the simulation orchestrator."""
    # --- 1. Argument Parsing: Define the experiment ---
    parser = argparse.ArgumentParser(
        description="Run the Project Genesis v16.0 simulation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Simulation setup
    parser.add_argument('-s', '--seed', type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument('-f', '--frames', type=int, default=1000, help="Max number of simulation frames.")

    # Topology setup
    parser.add_argument('--topo', type=str, default='crystal', choices=['crystal'], help="Type of substrate topology.")
    parser.add_argument('-W', '--width', type=int, default=80, help="Width for crystal topology.")
    parser.add_argument('-H', '--height', type=int, default=60, help="Height for crystal topology.")

    # Field and Initial Condition setup
    # For now, we only have a scalar field, so this choice is simple.
    # Later, we can add '--field_type spinor'.
    parser.add_argument('--ic', type=str, default='soup', choices=['soup', 'packet', 'vortex'], help="Initial condition type.")

    # Causality Computer setup
    parser.add_argument('-c', '--causality', type=str, default='conv', choices=['conv', 'div'], help="Causality computation rule (convergent, divergent).")

    args = parser.parse_args()

    # --- 2. Setup Run Environment ---
    SEED = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
    np.random.seed(SEED)

    # The field type is now part of the run name for clarity
    FIELD_TYPE = 'scalar' # Hardcoded for now
    run_name = f"SEED_{SEED}_{args.topo}{args.width}x{args.height}_{FIELD_TYPE}_{args.ic}_{args.causality}"
    RUN_DIR = f"run_{run_name}"
    DATA_DIR = os.path.join(RUN_DIR, 'data')

    cprint(f"\n--- PROJECT GENESIS v16.0: THE ENGINE ABSTRACTION ---", C.HEADER, attrs=C.BOLD_ATTR)
    cprint(f"Starting run: {run_name}", C.INFO)

    if os.path.exists(RUN_DIR):
        cprint(f"Warning: Run directory '{RUN_DIR}' already exists. Overwriting.", C.WARNING)
        shutil.rmtree(RUN_DIR)
    os.makedirs(DATA_DIR)

    # --- 3. Build Components (Dependency Injection Pattern) ---
    cprint("\n--- STAGE 1: ASSEMBLING COMPONENTS ---", C.SUBHEADER, attrs=C.BOLD_ATTR)

    # 3.1. Create the Substrate
    topo_params = {'width': args.width, 'height': args.height}
    topology_data = TopologyFactory.create(args.topo, topo_params)

    # 3.2. Create the appropriate Field object
    if FIELD_TYPE == 'scalar':
        field = ScalarField(topology_data.num_points)
    # elif FIELD_TYPE == 'spinor':
    #     field = SpinorField(topology_data.num_points)
    else:
        raise ValueError(f"Unknown field type: {FIELD_TYPE}")

    # 3.3. Use an Initial Condition generator to populate the field
    if args.ic == 'soup': initial_state_gen = PrimordialSoupState()
    elif args.ic == 'packet': initial_state_gen = WavePacketState()
    elif args.ic == 'vortex': initial_state_gen = VortexState()
    else: raise ValueError(f"Unknown initial condition: {args.ic}")

    # The generator now returns a modified field object
    field = initial_state_gen.generate(topology_data)

    # 3.4. Select the Causality Computer
    if args.causality == 'conv': causality_computer = ConvergentCausality()
    elif args.causality == 'div': causality_computer = DivergentCausality()
    else: raise ValueError(f"Unknown causality rule: {args.causality}")

    # 3.5. Inject all components into the Simulation Engine
    sim = Simulation(topology_data, field, causality_computer)

    # 3.6. Setup Analytics Tools
    tracker = ParticleTracker()
    analytics = GlobalAnalytics(sim.num_points)

    # --- 4. Store Metadata and Static Data ---
    substrate_path = os.path.join(RUN_DIR, "substrate.npz")
    np.savez_compressed(
        substrate_path,
        points=topology_data.points,
        neighbors=np.array(topology_data.neighbors, dtype=object)
    )

    metadata = {
        'run_name': run_name, 'seed': SEED, 'max_frames': args.frames,
        'field_type': FIELD_TYPE, # NEW: for the renderer
        'topology_type': args.topo, 'topology_params': topo_params,
        'initial_condition': args.ic, 'causality_rule': causality_computer.strategy_name,
        'final_frame_count': 0
    }

    cprint("\n--- STAGE 2: SIMULATING ---", C.SUBHEADER, attrs=C.BOLD_ATTR)
    start_time = time.time()
    final_frame_count = 0

    try:
        # --- 5. The Main Simulation Loop ---
        for frame in tqdm(range(args.frames), desc=f"Simulating ({args.causality} flow)", bar_format="{l_bar}{bar:30}{r_bar}"):
            causal_graph = sim.update_step()

            # The tracker and analytics modules are updated to work with the new field object
            interaction_source = sim.field.get_interaction_source()

            stable_attractors = tracker.analyze_frame(interaction_source, sim.field.values, sim.substrate, frame + 1)
            analytics.analyze_step(interaction_source, causal_graph, frame + 1)

            # Save the raw field values for the renderer
            frame_filename = os.path.join(DATA_DIR, f"frame_{frame+1:05d}.npz")
            np.savez_compressed(
                frame_filename,
                field_values=sim.field.values, # Save the full field data
                causal_graph=np.array(causal_graph, dtype=object),
                stable_attractors=np.array(stable_attractors, dtype=object)
            )
            final_frame_count = frame + 1

    except KeyboardInterrupt:
        cprint("\nSimulation interrupted by user.", C.WARNING)
    finally:
        sim.close()

    # --- 6. Finalize and Generate Reports ---
    cprint(f"\nSimulation finished at frame {final_frame_count}.", C.SUCCESS)
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds.")

    metadata['final_frame_count'] = final_frame_count
    metadata['global_interaction_source_max'] = analytics.global_interaction_source_max
    with open(os.path.join(RUN_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    analytics.generate_report(RUN_DIR)

    cprint(f"\nRun '{run_name}' complete.", C.HEADER, attrs=C.BOLD_ATTR)
    cprint(f"Data saved in '{RUN_DIR}'.\nTo render, run:\n"
           f"  python render_frames.py {RUN_DIR}", C.SUCCESS)

if __name__ == "__main__":
    main()
