# main.py v10.0 - "Simulator Core"
"""
Project Genesis - Simulation Core
-----------------------------------
- Теперь отвечает ТОЛЬКО за запуск симуляции и сохранение данных.
- Создает metadata.json с параметрами запуска.
- Рендеринг и компиляция вынесены в отдельные скрипты.
"""
import numpy as np
import argparse
import os
import shutil
import time
import json
from tqdm import tqdm
from termcolor import cprint

from topologies import CrystalTopology
from initial_conditions import WavePacketState, PrimordialSoupState, VortexState
from simulation import Simulation
from tracker import ParticleTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Project Genesis v10.0 simulation.")
    # ... (аргументы остаются теми же, но без --render-step и --keep-frames)
    parser.add_argument('-s', '--seed', type=int, help="Seed for reproducibility.")
    parser.add_argument('-w', '--width', type=int, default=80)
    parser.add_argument('-H', '--height', type=int, default=60)
    parser.add_argument('--ic', type=str, default='soup', choices=['packet', 'soup', 'vortex'])
    parser.add_argument('-f', '--frames', type=int, default=3000)
    parser.add_argument('-g', '--stable_goal', type=int, default=2)
    args = parser.parse_args()

    SEED = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
    np.random.seed(SEED)

    run_name = f"SEED_{SEED}_crystal_{args.width}x{args.height}_{args.ic}"
    DATA_DIR = f"run_{run_name}/data" # Новая структура папок

    cprint(f"\n--- STAGE 1: SIMULATION (IC: {args.ic.upper()}) ---", 'cyan')
    print(f"Using SEED: {SEED}")

    if os.path.exists(f"run_{run_name}"):
        cprint(f"Warning: Run directory 'run_{run_name}' already exists. Overwriting.", 'yellow')
        shutil.rmtree(f"run_{run_name}")
    os.makedirs(DATA_DIR)

    # --- СОЗДАНИЕ METADATA.JSON ---
    metadata = {
        'seed': SEED,
        'width': args.width,
        'height': args.height,
        'ic': args.ic,
        'frames_simulated': 0,
        'run_name': run_name
    }

    topology = CrystalTopology(width=args.width, height=args.height)
    # ... (выбор initial_state)
    if args.ic == 'packet': initial_state = WavePacketState()
    elif args.ic == 'soup': initial_state = PrimordialSoupState()
    else: initial_state = VortexState()

    sim = Simulation(topology, initial_state)
    tracker = ParticleTracker() # Используем трекер v3.0

    print("\nLaunching simulation...")
    start_time = time.time()
    final_frame_count = 0
    try:
        for frame in tqdm(range(args.frames), desc="Simulating"):
            sim.update_step()
            stable_particles = tracker.analyze_frame(sim, frame + 1)

            frame_filename = os.path.join(DATA_DIR, f"frame_{frame+1:05d}.npz")
            np.savez_compressed(
                frame_filename,
                points=sim.topology.points,
                psi=sim.psi,
                simplices=sim.simplices,
                stable_particles=np.array(stable_particles, dtype=object),
                tracked_count=len(tracker.tracked_particles)
            )
            final_frame_count = frame + 1

            if len(stable_particles) >= args.stable_goal:
                cprint(f"\nSUCCESS: Goal of {args.stable_goal} stable particle(s) reached at frame {frame + 1}.", 'green')
                break
    finally:
        sim.close()

    # Обновляем metadata.json с финальным числом кадров
    metadata['frames_simulated'] = final_frame_count
    with open(f"run_{run_name}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Simulation complete. Total time: {time.time() - start_time:.2f} seconds.")
    cprint(f"\nData saved in '{DATA_DIR}'.\nTo render, run:\n"
           f"python render_frames.py run_{run_name} -rs 10", 'green')
