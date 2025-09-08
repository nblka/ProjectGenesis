# test_renderer.py v2.0 - "Viewer"
"""
Утилита для рендеринга одного-единственного кадра из .npz файла.
---------------------------------------------------------------------
- Принимает путь к .npz файлу и путь к metadata.json как аргументы.
- Вызывает render_frame_worker для создания одного тестового кадра.
- Идеально подходит для быстрой отладки визуализации на конкретных,
  "проблемных" кадрах из реальной симуляции.

Пример Запуска:
python test_renderer.py run_SEED_12345.../data/frame_01234.npz run_SEED_12345.../metadata.json
"""
import numpy as np
import os
import json
import argparse
from termcolor import cprint

# --- Импортируем то, что тестируем ---
from renderer import render_frame_worker

def get_frame_number(path):
    """Извлекает номер кадра из имени файла."""
    try:
        return int(os.path.basename(path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        cprint(f"Warning: Could not extract frame number from '{path}'. Using 0.", "yellow")
        return 0

if __name__ == "__main__":
    # --- Настройка парсера ---
    parser = argparse.ArgumentParser(
        description="Render a single frame from a Project Genesis .npz data file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('data_file', type=str,
                        help="Path to the .npz data file to render (e.g., 'run_.../data/frame_00123.npz').")
    parser.add_argument('metadata_file', type=str,
                        help="Path to the corresponding metadata.json file (e.g., 'run_.../metadata.json').")
    args = parser.parse_args()

    DATA_PATH = args.data_file
    METADATA_PATH = args.metadata_file

    # --- Проверки на существование файлов ---
    if not os.path.exists(DATA_PATH):
        cprint(f"Error: Data file not found at '{DATA_PATH}'", 'red')
        exit()
    if not os.path.exists(METADATA_PATH):
        cprint(f"Error: Metadata file not found at '{METADATA_PATH}'", 'red')
        exit()

    cprint(f"--- Running Single Frame Renderer Test on '{os.path.basename(DATA_PATH)}' ---", "cyan")

    # --- Загружаем metadata ---
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        cprint(f"Error reading metadata file: {e}", "red")
        exit()

    # --- Готовим аргументы для render_frame_worker ---
    frame_num = get_frame_number(DATA_PATH)
    output_dir = "." # Сохраняем в текущую папку для простоты

    # Имя выходного файла будет информативным
    output_filename_base = f"test_render_SEED_{metadata.get('seed', 'unknown')}_frame_{frame_num}.png"

    # Убираем старый файл, если он есть
    if os.path.exists(output_filename_base):
        os.remove(output_filename_base)

    args_tuple = (
        frame_num,
        DATA_PATH,
        output_dir,
        metadata
    )

    # --- Вызываем рендерер! ---
    cprint("Calling render_frame_worker...", "yellow")
    # render_frame_worker ожидает, что имя файла будет frame_xxxxx.png
    temp_output_filename = os.path.join(output_dir, f"frame_{frame_num:05d}.png")

    result = render_frame_worker(args_tuple)

    # --- Проверяем результат и убираем за собой ---
    if result is None and os.path.exists(temp_output_filename):
        os.rename(temp_output_filename, output_filename_base)
        cprint(f"SUCCESS! Test frame saved as '{output_filename_base}'.", "green")
    else:
        cprint(f"FAILURE! Renderer returned an error: {result}", "red")
        if os.path.exists(temp_output_filename):
            os.remove(temp_output_filename) # Удаляем пустой или битый кадр

    cprint("--- Test Complete ---", "cyan")
