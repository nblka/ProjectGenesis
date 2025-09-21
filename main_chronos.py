# main_chronos.py v1.0
# Part of Project Chronos
# Orchestrator for running relaxation experiments.

import numpy as np
import argparse
import os
import shutil
import json
from termcolor import cprint

# --- Import Project Components ---
from topologies import TopologyData, generate_crystal_topology
from physics_law import DiscretizedDNLS
from relaxation_engine import RelaxationEngine
# We will need a new module for plotting/video later
# from reporting import generate_report 

# --- Helper Function to Load State from NPZ ---
def load_state_from_npz(filepath: str, num_points: int) -> np.ndarray:
    """Safely loads a psi state vector from a Project Genesis .npz frame file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"State file not found: {filepath}")
    
    with np.load(filepath, allow_pickle=True) as data:
        if 'field_values' not in data:
            raise KeyError(f"'field_values' not found in {filepath}")
        
        psi = data['field_values'].ravel()
        
        if len(psi) != num_points:
            raise ValueError(f"State in {filepath} has {len(psi)} points, but topology requires {num_points}.")
        
        return psi.astype(np.complex128)

def main():
    """Main function to configure and run a relaxation experiment."""
    parser = argparse.ArgumentParser(
        description="Project Chronos v1.0: A computational engine for finding self-consistent histories.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Input Configuration ---
    parser.add_argument('--start_file', type=str, default=None,
                        help="Path to a .npz file for the initial state (t=0).")
    parser.add_argument('--end_file', type=str, default=None,
                        help="Path to a .npz file for the target final state (t=T).")
    parser.add_argument('--steps', type=int, default=100,
                        help="Number of time steps for the history between start and end.")
    parser.add_argument('--topo_width', type=int, default=20, help="Width of the crystal substrate.")
    parser.add_argument('--topo_height', type=int, default=1, help="Height of the crystal substrate (1 for 1D).")
    
    # --- Law Configuration ---
    parser.add_argument('--alpha', type=float, default=1.0, help="Initial guess for the non-linearity parameter alpha.")
    parser.add_argument('--dt', type=float, default=0.01, help="Time step for the physical law.")

    # --- Training Configuration ---
    parser.add_argument('--iterations', type=int, default=2000, help="Number of relaxation iterations.")
    parser.add_argument('--lr_law', type=float, default=0.01, help="Learning rate for the physical law parameters.")
    parser.add_argument('--lr_hist', type=float, default=0.001, help="Learning rate for the history relaxation.")
    
    # --- Output Configuration ---
    parser.add_argument('--run_name', type=str, default=f"chronos_run_{np.random.randint(1000, 9999)}",
                        help="A unique name for this experiment run.")

    args = parser.parse_args()

    # === 1. SETUP ENVIRONMENT ===
    cprint(f"\n--- PROJECT CHRONOS: Initializing Run '{args.run_name}' ---", "magenta", attrs=['bold'])
    
    run_dir = f"run_{args.run_name}"
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    # === 2. BUILD THE SUBSTRATE (TOPOLOGY) ===
    cprint("1. Building substrate...", "cyan")
    topology = generate_crystal_topology(width=args.topo_width, height=args.topo_height)
    num_points = topology.num_points

    # === 3. LOAD OR CREATE BOUNDARY STATES ===
    cprint("2. Preparing boundary conditions...", "cyan")
    
    if args.start_file:
        cprint(f"   -> Loading start state from: {args.start_file}", "white")
        start_state = load_state_from_npz(args.start_file, num_points)
    else:
        cprint("   -> No start file provided, creating a default wave packet.", "yellow")
        x = np.arange(num_points)
        start_state = np.exp(-(x - num_points*0.2)**2 / 20)
        start_state /= np.linalg.norm(start_state)

    if args.end_file:
        cprint(f"   -> Loading end state from: {args.end_file}", "white")
        end_state = load_state_from_npz(args.end_file, num_points)
    else:
        cprint("   -> No end file provided, creating a default wave packet.", "yellow")
        x = np.arange(num_points)
        end_state = np.exp(-(x - num_points*0.8)**2 / 20)
        end_state /= np.linalg.norm(end_state)

    # === 4. INITIALIZE THE ENGINE ===
    cprint("3. Assembling the Relaxation Engine...", "cyan")
    initial_law = DiscretizedDNLS(alpha=args.alpha, dt=args.dt)
    
    engine = RelaxationEngine(
        start_state=start_state,
        end_state_target=end_state,
        num_steps=args.steps,
        initial_law=initial_law,
        topology=topology
    )

    # === 5. RUN THE TRAINING ===
    cprint("4. Starting relaxation and learning process...", "cyan", attrs=['bold'])
    optimal_law, optimal_history = engine.relax_and_learn(
        iterations=args.iterations,
        lr_law=args.lr_law,
        lr_history=args.lr_hist
    )

    # === 6. SAVE RESULTS ===
    cprint("5. Saving results...", "cyan")
    
    # Save the final learned law
    law_params = {'alpha': optimal_law.alpha, 'dt': optimal_law.dt}
    with open(os.path.join(run_dir, 'optimal_law.json'), 'w') as f:
        json.dump(law_params, f, indent=4)
        
    # Save the full, self-consistent history
    np.savez_compressed(os.path.join(run_dir, 'optimal_history.npz'), history=optimal_history)
    
    # Save the metadata for reproducibility
    metadata = vars(args) # Converts argparse Namespace to a dict
    with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    cprint(f"   -> Results saved to directory: '{run_dir}'", "green")
    
    # === 7. GENERATE REPORT (Future Step) ===
    cprint("6. Generating visual report (TODO)...", "yellow")
    # generate_report(run_dir)
    
    cprint("\n--- Run Complete ---", "magenta", attrs=['bold'])


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as e:
        cprint(f"\nFATAL ERROR: {e}", "red", attrs=['bold'])
        exit(1)