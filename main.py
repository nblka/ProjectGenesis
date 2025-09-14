# main.py v15.0
# Part of Project Genesis: Breathing Causality
# v15.0: Final, robust version of the simulation orchestrator.
# - Integrates all v13+ modular, data-centric components.
# - Centralizes parameter handling via argparse.
# - Manages the full lifecycle: Setup -> Simulation -> Analytics.
# - Produces a self-contained run directory with all data and metadata.
# - Saves all tracked particles per frame for enhanced visualization.

import numpy as np
import argparse
import os
import shutil
import time
import json
from tqdm import tqdm
from termcolor import cprint

# --- Import all final, tested components ---
from topologies import TopologyFactory
from causality import AmplitudeConvergentCausality, AmplitudeDivergentCausality
from initial_conditions import PrimordialSoupState, WavePacketState, VortexState
from simulation import Simulation
from tracker import ParticleTracker
from analytics import GlobalAnalytics

def main():
    """Main function to run the simulation orchestrator."""
    # --- 1. Argument Parsing: Define the experiment ---
    parser = argparse.ArgumentParser(
        description="Run the Project Genesis v15.0 simulation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Simulation setup
    parser.add_argument('-s', '--seed', type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument('-f', '--frames', type=int, default=1000, help="Max number of simulation frames.")

    # Topology setup
    parser.add_argument('--topo', type=str, default='crystal', choices=['crystal'], help="Type of substrate topology.")
    parser.add_argument('-w', '--width', type=int, default=80, help="Width for crystal topology.")
    parser.add_argument('-H', '--height', type=int, default=60, help="Height for crystal topology.")

    # Initial Condition setup
    parser.add_argument('--ic', type=str, default='soup', choices=['soup', 'packet', 'vortex'], help="Initial condition type.")

    # Causality Evolver setup
    parser.add_argument('--causality', type=str, default='convergent', choices=['convergent', 'divergent'], help="Causality flow rule.")

    args = parser.parse_args()

    # --- 2. Setup Run Environment ---
    SEED = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
    np.random.seed(SEED)

    run_name = f"SEED_{SEED}_{args.topo}{args.width}x{args.height}_{args.ic}_{args.causality}"
    RUN_DIR = f"run_{run_name}"
    DATA_DIR = os.path.join(RUN_DIR, 'data')

    cprint(f"\n--- PROJECT GENESIS v15.0: BREATHING CAUSALITY ---", 'magenta', attrs=['bold'])
    cprint(f"Starting run: {run_name}", 'white')

    if os.path.exists(RUN_DIR):
        cprint(f"Warning: Run directory '{RUN_DIR}' already exists. Overwriting.", 'yellow')
        shutil.rmtree(RUN_DIR)
    os.makedirs(DATA_DIR)

    # --- 3. Build Components (Dependency Injection) ---

    # 3.1. Topology
    topo_params = {'width': args.width, 'height': args.height}
    topology_data = TopologyFactory.create(args.topo, topo_params)

    # 3.2. Initial Condition
    if args.ic == 'soup': initial_state_gen = PrimordialSoupState()
    elif args.ic == 'packet': initial_state_gen = WavePacketState()
    elif args.ic == 'vortex': initial_state_gen = VortexState()
    else: raise ValueError(f"Unknown initial condition: {args.ic}")

    # 3.3. Causality Evolver
    if args.causality == 'convergent': causality_evolver = AmplitudeConvergentCausality()
    elif args.causality == 'divergent': causality_evolver = AmplitudeDivergentCausality()
    else: raise ValueError(f"Unknown causality rule: {args.causality}")

    # 3.4. Assemble the Simulation
    sim = Simulation(topology_data, causality_evolver, initial_state_gen)

    # 3.5. Setup Analytics Tools
    tracker = ParticleTracker()
    analytics = GlobalAnalytics(sim.num_points)

    # --- 4. Store Comprehensive Metadata and Static Data ---

    # 4.1. Save HEAVY static data to its own file
    substrate_path = os.path.join(RUN_DIR, "substrate.npz")
    np.savez_compressed(
        substrate_path,
        points=topology_data.points,
        neighbors=np.array(topology_data.neighbors, dtype=object)
    )

    # 4.2. Save LIGHT metadata to JSON
    metadata = {
        'run_name': run_name, 'seed': SEED, 'max_frames': args.frames,
        'topology_type': args.topo, 'topology_params': topo_params,
        'initial_condition': args.ic, 'causality_rule': causality_evolver.strategy_name,
        'final_frame_count': 0
    }

    cprint("\n--- STAGE 1: SIMULATING ---", 'cyan')
    start_time = time.time()
    final_frame_count = 0

    try:
        # --- 5. The Main Simulation Loop ---
        for frame in tqdm(range(args.frames), desc=f"Simulating ({args.causality} flow)"):
            causal_graph = sim.update_step()
            stable_attractors = tracker.analyze_frame(sim, frame + 1)
            all_tracked_particles = list(tracker.tracked_particles.values())
            analytics.analyze_step(sim.psi, causal_graph, frame + 1)

            frame_filename = os.path.join(DATA_DIR, f"frame_{frame+1:05d}.npz")
            np.savez_compressed(
                frame_filename,
                psi=sim.psi,
                causal_graph=np.array(causal_graph, dtype=object),
                all_tracked_particles=np.array(all_tracked_particles, dtype=object),
                stable_attractors=np.array(stable_attractors, dtype=object),
                tracked_count=len(tracker.tracked_particles)
            )
            final_frame_count = frame + 1

    except KeyboardInterrupt:
        cprint("\nSimulation interrupted by user.", 'yellow')
    finally:
        sim.close()

    # --- 6. Finalize and Generate Reports ---
    cprint(f"\nSimulation finished at frame {final_frame_count}.", 'green')
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds.")

    metadata['final_frame_count'] = final_frame_count
    with open(os.path.join(RUN_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    analytics.generate_report(RUN_DIR)

    cprint(f"\nRun '{run_name}' complete.", 'magenta', attrs=['bold'])
    cprint(f"Data saved in '{DATA_DIR}'.\nTo render, run:\n"
           f"  python render_frames.py {RUN_DIR}", 'green')
    cprint(f"To compile the video, run:\n"
           f"  python compile_video.py {RUN_DIR}", 'green')

if __name__ == "__main__":
    main()
