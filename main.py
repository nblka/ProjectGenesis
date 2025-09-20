# main.py v19.0
# Part of Project Genesis: Breathing Causality
# v19.0: "Research Mode"
# - Adds a new `--fast-mode` command-line argument. When enabled, the simulation
#   skips the computationally expensive saving of per-frame data files.
# - This is ideal for quickly scanning parameter spaces, where only the final
#   analytics report (the graph) is needed.
# - The interaction strength is now a `base_strength` parameter, reflecting
#   the new physics in the simulation engine.

import numpy as np
import argparse
import os
import shutil
import time
import json
from tqdm import tqdm

# --- Import all components (no changes in imports) ---
from styling import C, cprint
from topologies import TopologyFactory
from field import ScalarField
from initial_conditions import PrimordialSoupState, WavePacketState, VortexState, SeededSoupState
from causality import ConvergentCausality, DivergentCausality
from simulation import Simulation
from tracker import ParticleTracker
from analytics import GlobalAnalytics

def main():
    """Main function to run the simulation orchestrator."""
    parser = argparse.ArgumentParser(description="Run the Project Genesis v19.0 simulation.")
    
    parser.add_argument('-s', '--seed', type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument('-f', '--frames', type=int, default=1000, help="Max number of simulation frames.")
    parser.add_argument('--topo', type=str, default='crystal', choices=['crystal'], help="Type of substrate topology.")
    parser.add_argument('-W', '--width', type=int, default=80, help="Width for crystal topology.")
    parser.add_argument('-H', '--height', type=int, default=60, help="Height for crystal topology.")
    parser.add_argument('--ic', type=str, default='soup', 
                        choices=['soup', 'seeded', 'packet', 'vortex'], 
                        help="Initial condition type.")
    parser.add_argument('--seed_strength', type=float, default=5.0, help="Strength of the seed for 'seeded' IC.")
    parser.add_argument('--seed_size', type=int, default=1, help="Radius of the seed for 'seeded' IC.")
    parser.add_argument('-c', '--causality', type=str, default='conv', choices=['conv', 'div'], help="Causality computation rule.")
    parser.add_argument('-is', '--strength', type=float, default=200.0, help="Base interaction strength for the dynamic potential.")
    parser.add_argument('--fast', action='store_true', help="Enable fast mode: skip saving per-frame data to disk.")
    
    args = parser.parse_args()

    # --- Setup Run Environment (updated run_name) ---
    SEED = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
    np.random.seed(SEED)

    FIELD_TYPE = 'scalar' # Hardcoded for now

    mode_str = "FAST" if args.fast else "FULL"
    ic_str = args.ic
    if args.ic == 'seeded':
        ic_str = f"seeded{args.seed_strength}s{args.seed_size}r"
    run_name = f"SEED_{SEED}_{args.topo}{args.width}x{args.height}_s{args.strength}_{ic_str}_{args.causality}"
    RUN_DIR = f"run_{run_name}"
    DATA_DIR = os.path.join(RUN_DIR, 'data')
    
    cprint(f"\n--- PROJECT GENESIS v19.0: LOCAL CATALYSIS ---", C.HEADER, attrs=C.BOLD_ATTR)
    cprint(f"Starting run: {run_name}", C.INFO)

    if os.path.exists(RUN_DIR):
        cprint(f"Warning: Run directory '{RUN_DIR}' already exists. Overwriting.", C.WARNING)
        shutil.rmtree(RUN_DIR)
    os.makedirs(DATA_DIR)
    
    # --- Build Components ---
    cprint("\n--- STAGE 1: ASSEMBLING COMPONENTS ---", C.SUBHEADER, attrs=C.BOLD_ATTR)
    
    topo_params = {'width': args.width, 'height': args.height}
    topology_data = TopologyFactory.create(args.topo, topo_params)
    
    if args.ic == 'soup': 
        initial_state_gen = PrimordialSoupState()
    elif args.ic == 'seeded':
        initial_state_gen = SeededSoupState(seed_strength=args.seed_strength, seed_size=args.seed_size)
    elif args.ic == 'packet': 
        initial_state_gen = WavePacketState()
    elif args.ic == 'vortex': 
        initial_state_gen = VortexState()
    else: 
        raise ValueError(f"Unknown initial condition: {args.ic}")
    
    if FIELD_TYPE == 'scalar':
        field = ScalarField(topology_data.num_points)
    # elif FIELD_TYPE == 'spinor':
    #     field = SpinorField(topology_data.num_points)    if args.causality == 'conv': causality_computer = ConvergentCausality()
    else: causality_computer = DivergentCausality()

    # Inject the new base_interaction_strength parameter
    sim = Simulation(topology_data, field, causality_computer, base_interaction_strength=args.strength)

    analytics = GlobalAnalytics(sim.num_points)
    # Tracker can be skipped in fast mode
    tracker = None if args.fast else ParticleTracker()

    # --- Metadata (unchanged) ---
    substrate_path = os.path.join(RUN_DIR, "substrate.npz")
    np.savez_compressed(
        substrate_path,
        points=topology_data.points,
        neighbors=np.array(topology_data.neighbors, dtype=object)
    )
    metadata = { 'run_name': run_name, 'seed': SEED, 'max_frames': args.frames, 'base_interaction_strength': args.strength,
                 'field_type': 'scalar', 'topology_type': args.topo, 'topology_params': topo_params, 'initial_condition': args.ic,
                 'causality_rule': causality_computer.strategy_name, 'final_frame_count': 0 }

    cprint("\n--- STAGE 2: SIMULATING ---", C.SUBHEADER, attrs=C.BOLD_ATTR)
    start_time = time.time()
    final_frame_count = 0

    try:
        # --- The Main Simulation Loop ---
        for frame in tqdm(range(args.frames), desc=f"Simulating ({args.causality} flow)", bar_format="{l_bar}{bar:30}{r_bar}"):
            causal_graph = sim.update_step()
            
            # Analytics are always computed as they are lightweight
            interaction_source = sim.field.get_interaction_source()
            analytics.analyze_step(interaction_source, causal_graph, frame + 1)

            # --- Conditional saving ---
            if not args.fast:
                # Particle tracking and saving is only done in full mode
                stable_attractors = tracker.analyze_frame(interaction_source, sim.field.values, sim.substrate, frame + 1)
                
                frame_filename = os.path.join(DATA_DIR, f"frame_{frame+1:05d}.npz")
                np.savez_compressed(
                    frame_filename,
                    field_values=sim.field.values,
                    causal_graph=np.array(causal_graph, dtype=object),
                    stable_attractors=np.array(stable_attractors, dtype=object)
                )
            final_frame_count = frame + 1

    except KeyboardInterrupt:
        cprint("\nSimulation interrupted by user.", C.WARNING)
    finally:
        sim.close()
        
    # --- Finalize and Generate Reports ---
    cprint(f"\nSimulation finished at frame {final_frame_count}.", C.SUCCESS)
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds.")

    metadata['final_frame_count'] = final_frame_count
    metadata['global_interaction_source_max'] = analytics.global_interaction_source_max
    with open(os.path.join(RUN_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    analytics.generate_report(RUN_DIR)
    
    cprint(f"\nRun '{run_name}' complete.", C.HEADER, attrs=C.BOLD_ATTR)
    if not args.fast:
        cprint(f"Data saved in '{RUN_DIR}'. To render, run:\n"
               f"  python render_frames.py {RUN_DIR}", C.SUCCESS)
    else:
        cprint(f"Ran in fast mode. Analytics saved in '{os.path.join(RUN_DIR, 'analytics')}'. No frames to render.", C.SUCCESS)

if __name__ == "__main__":
    main()