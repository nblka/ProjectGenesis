# render_frames.py v10.1
"""
Project Genesis - Frame Rendering Module
----------------------------------------
- Самостоятельный скрипт для рендеринга кадров из папки с данными.
- Читает metadata.json для получения информации о симуляции.
- Поддерживает прореживание кадров (--render-step).
- v10.1: ИСПРАВЛЕНА ошибка, из-за которой размер узлов не зависел от амплитуды.
         Возвращена правильная формула для node_sizes и улучшен zorder.
"""
import numpy as np
import argparse
import os
import glob
import shutil
import json
import multiprocessing as mp
from tqdm import tqdm
from termcolor import cprint

# --- Импортируем сам воркер из renderer.py, чтобы сохранить модульность ---
# (Убедитесь, что файл renderer.py с правильным кодом воркера существует)
from renderer import render_frame_worker

def get_frame_number(path):
    """Извлекает номер кадра из имени файла для сортировки и именования."""
    try:
        # Извлекаем номер из '.../frame_00123.npz'
        return int(os.path.basename(path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        # Если имя файла не соответствует шаблону, возвращаем -1 для игнорирования
        return -1

if __name__ == "__main__":
    # Устанавливаем метод запуска мультипроцессинга (важно для Windows/macOS)
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Render frames from a Project Genesis simulation run.")
    parser.add_argument('run_directory', type=str, help="Path to the run directory (e.g., 'run_SEED_12345...').")
    parser.add_argument('-rs', '--render-step', type=int, default=1, help="Render every N-th frame.")
    args = parser.parse_args()

    RUN_DIR = args.run_directory
    DATA_DIR = os.path.join(RUN_DIR, 'data')
    FRAMES_DIR = os.path.join(RUN_DIR, 'frames')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')

    if not os.path.exists(METADATA_PATH):
        cprint(f"Error: metadata.json not found in '{RUN_DIR}'. Make sure you provide the correct run directory.", 'red'); exit()

    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        cprint(f"Error: Could not read or parse metadata.json in '{RUN_DIR}'.", 'red'); exit()


    cprint(f"\n--- STAGE 2: RENDERING FRAMES for run '{metadata['run_name']}' ---", 'cyan')

    if os.path.exists(FRAMES_DIR):
        cprint(f"Warning: Frames directory '{FRAMES_DIR}' already exists. Overwriting.", 'yellow')
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR)

    all_data_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npz')), key=get_frame_number)

    if not all_data_files:
        cprint(f"Error: No data files (.npz) found in '{DATA_DIR}'.", 'red'); exit()

    # Фильтрация кадров по шагу
    data_files_to_render = [p for p in all_data_files if get_frame_number(p) % args.render_step == 0 and get_frame_number(p) != -1]

    cprint(f"Selected {len(data_files_to_render)} of {len(all_data_files)} data files with step {args.render_step}.", 'yellow')

    # Создаем список задач для пула процессов
    tasks = []
    for path in data_files_to_render:
        frame_num = get_frame_number(path)
        # Передаем РЕАЛЬНЫЙ номер кадра в воркер для корректного отображения в заголовке
        tasks.append((frame_num, path, FRAMES_DIR, metadata))

    if tasks:
        # Используем менеджер контекста для пула, чтобы он корректно закрывался
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(render_frame_worker, tasks), total=len(tasks), desc="Rendering"))

        failed_frames = [res for res in results if res is not None]
        if failed_frames:
            cprint(f"\nWarning: {len(failed_frames)} frame(s) failed to render. Check logs for details.", 'yellow')

    cprint("Rendering complete.", 'green')
    cprint(f"\nFrames saved in '{FRAMES_DIR}'.\nTo compile the video, run:\n"
           f"python compile_video.py {RUN_DIR}", 'green')
