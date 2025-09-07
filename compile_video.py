# compile_video.py v10.3 - "Absolute Path"
"""
Project Genesis - Video Compilation Module
------------------------------------------
- v10.3:
  - Теперь использует АБСОЛЮТНЫЕ пути для максимальной надежности.
  - Решает проблему 'Impossible to open' на всех платформах.
  - Упрощена обработка имен файлов.
"""
import subprocess
import os
import shutil
import json
import argparse
import glob
from termcolor import cprint

def get_frame_number_for_sort(path):
    """Извлекает номер кадра из имени файла для корректной сортировки."""
    try:
        return int(os.path.basename(path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        return -1

def compile_video_worker(frames_dir, output_filename, framerate=30, keep_frames=True):
    """Воркер-функция, использующая concat демуксер с абсолютными путями."""

    # ИСПРАВЛЕНИЕ 1: Сразу получаем абсолютный путь к папке с кадрами
    abs_frames_dir = os.path.abspath(frames_dir)

    if not os.path.exists(abs_frames_dir) or not os.listdir(abs_frames_dir):
        cprint(f"Error: Frames directory '{abs_frames_dir}' is empty or does not exist.", 'red')
        return

    cprint("Searching for frame files...", 'yellow')
    # glob вернет пути относительно текущей папки, но мы их преобразуем
    frame_files_relative = glob.glob(os.path.join(frames_dir, 'frame_*.png'))
    if not frame_files_relative:
        cprint(f"Error: No .png frames found in '{frames_dir}'.", 'red')
        return

    frame_files_relative.sort(key=get_frame_number_for_sort)

    # Преобразуем все найденные пути в абсолютные
    frame_files_absolute = [os.path.abspath(f) for f in frame_files_relative]

    list_filename = os.path.join(abs_frames_dir, "file_list.txt")

    # <<< ОТЛАДОЧНАЯ ПЕЧАТЬ >>>
    cprint(f"Found {len(frame_files_absolute)} frames. The first 3 are:", 'magenta')
    for i in range(min(3, len(frame_files_absolute))):
        cprint(f"  - {frame_files_absolute[i]}", 'magenta')
    # <<< КОНЕЦ ОТЛАДКИ >>>

    try:
        with open(list_filename, 'w', encoding='utf-8') as f:
            for filename in frame_files_absolute:
                # ИСПРАВЛЕНИЕ 2: Упрощенный и более надежный формат записи
                # Просто заменяем обратные слэши на прямые, этого достаточно
                safe_path = filename.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")

        # ffmpeg будет запущен из текущей папки, но будет работать с абсолютными путями,
        # что устраняет любую неоднозначность.
        # --- Собираем НОВУЮ, БОЛЕЕ НАДЕЖНУЮ команду ffmpeg ---
        ffmpeg_command = [
            'ffmpeg',
            '-y',                  # Перезаписывать без вопроса (лучше ставить в начале)
            '-f', 'concat',
            '-safe', '0',
            '-r', str(framerate),  # <<< ЯВНО УКАЗЫВАЕМ ЧАСТОТУ КАДРОВ ДЛЯ ВХОДНОГО ПОТОКА
            '-i', list_filename,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', # <<< ИСПРАВЛЕНА ОПЕЧАТКА
            '-r', str(framerate),  # <<< ЯВНО УКАЗЫВАЕМ ЧАСТОТУ КАДРОВ ДЛЯ ВЫХОДНОГО ПОТОКА
            os.path.abspath(output_filename)
        ]

        cprint(f"Running FFMPEG with concat demuxer...", 'yellow')
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True, encoding='utf-8')

        cprint(f"Video compilation successful! Output saved to '{os.path.abspath(output_filename)}'", 'green')

        if not keep_frames:
            cprint(f"Cleaning up frames directory '{frames_dir}'...", 'yellow')
            shutil.rmtree(frames_dir)

    except subprocess.CalledProcessError as e:
        cprint(f"\n--- FFMPEG ERROR ---", 'red')
        cprint(f"FFMPEG failed. The command was:\n{' '.join(e.cmd)}", 'yellow')
        cprint(f"\nFFMPEG stderr output:\n{e.stderr}", 'red')
        cprint("PNG frames were NOT deleted.", 'cyan')
    except Exception as e:
        cprint(f"\nAn unexpected error occurred: {e}", 'red')
    finally:
        if os.path.exists(list_filename):
            os.remove(list_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile a video from Project Genesis frames.")
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('-fr', '--framerate', type=int, default=30, help="Framerate of the output video.")
    parser.add_argument('--delete-frames', action='store_true', help="Delete the PNG frames folder after successful compilation.")
    args = parser.parse_args()

    RUN_DIR = args.run_directory
    FRAMES_DIR = os.path.join(RUN_DIR, 'frames')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')

    if not os.path.exists(METADATA_PATH):
        cprint(f"Error: metadata.json not found in '{RUN_DIR}'.", 'red'); exit()
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    cprint(f"\n--- STAGE 3: COMPILING VIDEO for run '{metadata['run_name']}' ---", 'cyan')

    output_filename = os.path.join(RUN_DIR, f"genesis_{metadata['run_name']}_v10.3.mp4")
    keep_frames_flag = not args.delete_frames

    compile_video_worker(FRAMES_DIR, output_filename, args.framerate, keep_frames_flag)
    cprint("--- Compilation Complete ---", 'cyan')
