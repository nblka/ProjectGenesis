# run_simulation.py
"""
Project Genesis v5.0 - Simulation Core
---------------------------------------
This script ONLY runs the simulation and saves the raw data for each frame.
It does NOT perform any visualization to ensure maximum stability.
"""
import numpy as np
import argparse
import time
import os
import multiprocessing as mp

from simulation import WavePacketUniverse, PrimordialSoupUniverse
from tracker import ParticleTracker

parser = argparse.ArgumentParser(description="Run the Project Genesis simulation (data generation only).")
# ... (все аргументы остаются прежними)
parser.add_argument('--seed', type=int, help="Specify a seed for reproducibility. If not given, a random one is used.")
parser.add_argument('--points', type=int, default=500, help="Number of points.")
parser.add_argument('--mode', type=str, default='packet', choices=['packet', 'soup'], help=("'packet': Start with a single, coherent wave packet.\n" "'soup': Start from a high-energy, random primordial soup."))
parser.add_argument('--frames', type=int, default=2000, help="Maximum number of frames.")
parser.add_argument('--stable_particles', type=int, default=1, help="Target number of stable particles to find.")
args = parser.parse_args()
SEED = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
np.random.seed(SEED)
USE_MULTIPROCESSING = mp.cpu_count() > 1

if __name__ == "__main__":
    print("\n--- Project Genesis v5.0 (Simulation Stage) ---")
    print(f"Using SEED: {SEED}")

    DATA_DIR = f"data_SEED_{SEED}"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    print(f"Raw data will be saved in: '{DATA_DIR}'")

    sim = WavePacketUniverse(num_points=args.points, use_multiprocessing=USE_MULTIPROCESSING) if args.mode == 'packet' else PrimordialSoupUniverse(num_points=args.points, use_multiprocessing=USE_MULTIPROCESSING, initial_energy=5.0)
    tracker = ParticleTracker(pool=sim.pool)

    print("4. Launching simulation...")
    start_time = time.time()

    try:
        for frame in range(args.frames):
            sim.update_step()
            stable_particles = tracker.analyze_frame(sim, frame + 1)

            # Save all necessary data for this frame
            frame_filename = os.path.join(DATA_DIR, f"frame_{frame:05d}.npz")
            np.savez_compressed(
                frame_filename,
                points=sim.points,
                psi=sim.psi,
                simplices=sim.simplices,
                stable_particles=stable_particles,
                tracked_count=len(tracker.tracked_particles)
            )

            if (frame + 1) % 50 == 0:
                print(f"  ... Simulated frame {frame + 1}/{args.frames}")

            if len(stable_particles) >= args.stable_particles:
                print(f"\nSUCCESS: Goal of {args.stable_particles} stable particle(s) reached at frame {frame + 1}.")
                break
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
    finally:
        sim.close_pool()
        end_time = time.time()
        print("\nSimulation complete.")
        print(f"Total simulation time: {end_time - start_time:.2f} seconds.")
        print(f"Data saved in '{DATA_DIR}'. Now run 'render_video.py' to create the animation.")