# main.py
"""
Project Genesis - Main Control Script ('Crystal')
------------------------------------------------------
- Управляет полным циклом симуляции на идеальной кристаллической решетке.
- Использует модульную архитектуру:
  - `topologies.py` для создания геометрии
  - `initial_conditions.py` для создания поля psi
  - `simulation.py` для запуска физического движка
  - `tracker.py` для анализа результатов
  - `renderer.py` и `compiler.py` для визуализации
"""
import numpy as np
import argparse
import os
import glob
import shutil
import time
import multiprocessing as mp
from tqdm import tqdm
from termcolor import cprint

# --- Импорт наших модулей ---
# (Убедитесь, что все эти файлы находятся в той же папке)
from topologies import CrystalTopology
from initial_conditions import WavePacketState, PrimordialSoupState
from simulation import Simulation
from tracker import ParticleTracker
from renderer import render_frame_worker
from compiler import compile_video

# --- Основная логика ---
if __name__ == "__main__":
    # Устанавливаем метод запуска мультипроцессинга (важно для Windows)
    mp.set_start_method('spawn', force=True)

    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Run the Project Genesis v9.0 'Crystal' simulation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--seed', type=int, help="Seed for reproducibility. Random if not given.")
    parser.add_argument('--width', type=int, default=80, help="Width of the crystal lattice.")
    parser.add_argument('--height', type=int, default=60, help="Height of the crystal lattice.")
    parser.add_argument('--ic', type=str, default='soup', choices=['packet', 'soup'],
                        help="'packet': Start with a single, coherent wave packet.\n"
                             "'soup': Start from a high-energy, random primordial soup.")
    parser.add_argument('--frames', type=int, default=1500, help="Maximum number of frames to simulate.")
    parser.add_argument('--stable_goal', type=int, default=5, help="Target number of stable particles to find.")
    parser.add_argument('--keep-data', action='store_true', help="Keep the raw NPZ data files after completion.")
    parser.add_argument('--keep-frames', action='store_true', help="Keep the PNG frame files after rendering.")
    args = parser.parse_args()

    # --- Инициализация ---
    SEED = args.seed if args.seed is not None else np.random.randint(0, 1_000_000)
    np.random.seed(SEED)

    run_name = f"SEED_{SEED}_crystal_{args.width}x{args.height}_{args.ic}"
    DATA_DIR = f"data_{run_name}"
    FRAMES_DIR = f"frames_{run_name}"
    OUTPUT_FILENAME = f'genesis_{run_name}_v9.0.mp4'

    # --- STAGE 1: SIMULATION ---
    cprint("\n--- STAGE 1: SIMULATION (Crystal Lattice) ---", 'cyan')
    print(f"Using SEED: {SEED}")
    if os.path.exists(DATA_DIR):
        cprint(f"Warning: Data directory '{DATA_DIR}' already exists. Overwriting.", 'yellow')
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    # Собираем нашу 'вселенную' как конструктор!
    topology = CrystalTopology(width=args.width, height=args.height)
    initial_state = WavePacketState() if args.ic == 'packet' else PrimordialSoupState()
    sim = Simulation(topology, initial_state)
    tracker = ParticleTracker()

    print("\n3. Launching simulation...")
    start_time = time.time()
    try:
        for frame in tqdm(range(args.frames), desc="Simulating"):
            sim.update_step()
            stable_particles = tracker.analyze_frame(sim, frame + 1)

            frame_filename = os.path.join(DATA_DIR, f"frame_{frame:05d}.npz")
            np.savez_compressed(
                frame_filename,
                points=sim.topology.points,
                psi=sim.psi,
                simplices=sim.simplices,
                stable_particles=np.array(stable_particles, dtype=object),
                tracked_count=len(tracker.tracked_particles)
            )

            if len(stable_particles) >= args.stable_goal:
                cprint(f"\nSUCCESS: Goal of {args.stable_goal} stable particle(s) reached at frame {frame + 1}.", 'green')
                break
    finally:
        sim.close() # Важно для корректного завершения дочерних процессов

    print(f"Simulation complete. Total time: {time.time() - start_time:.2f} seconds.")

    # --- STAGE 2: RENDERING ---
    cprint("\n--- STAGE 2: RENDERING FRAMES ---", 'cyan')
    if not os.path.exists(FRAMES_DIR): os.makedirs(FRAMES_DIR)
    data_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npz')))
    if not data_files:
        cprint(f"Error: No data found in '{DATA_DIR}'.", 'red'); exit()

    shared_info = {'seed': SEED, 'topology': f"{args.width}x{args.height}", 'ic': args.ic, 'width': args.width, 'height': args.height}
    tasks = [(i, path, FRAMES_DIR, shared_info) for i, path in enumerate(data_files)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(render_frame_worker, tasks), total=len(tasks), desc="Rendering"))

    failed_frames = [res for res in results if res is not None]
    if failed_frames:
        cprint(f"\nWarning: {len(failed_frames)} frame(s) failed to render.", 'yellow')

    # --- STAGE 3: COMPILATION ---
    compile_video(FRAMES_DIR, OUTPUT_FILENAME, keep_frames=args.keep_frames)

    # --- Финальная очистка ---
    if not args.keep_data and os.path.exists(DATA_DIR):
        print(f"Cleaning up temporary data directory '{DATA_DIR}'...")
        shutil.rmtree(DATA_DIR)

    cprint("\nProject Genesis v9.0 run complete.", 'cyan')
