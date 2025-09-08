# compile_video.py v10.5 - "Timestamp Perfect"
"""
Project Genesis - Video Compilation Module
------------------------------------------
- v10.5:
  - Добавлена директива 'duration' в file_list.txt для concat demuxer.
  - Это решает проблему с отбрасыванием кадров (dropping frames) и
    неправильными таймстампами (PTS/DTS) в FFMPEG.
  - Упрощена и сделана более надежной основная команда ffmpeg.
"""
import subprocess
import os
import shutil
import json
import argparse
import glob
from termcolor import cprint

def get_frame_number_for_sort(path):
    """Извлекает номер кадра из имени файла для корректной числовой сортировки."""
    try:
        return int(os.path.basename(path).split('_')[-1].split('.')[0])
    except (IndexError, ValueError):
        return -1

def compile_video_worker(frames_dir, output_filename, framerate=30, keep_frames=False):
    """Воркер-функция, использующая concat demuxer с явным указанием длительности кадров."""

    if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
        cprint(f"Error: Frames directory '{frames_dir}' not found.", 'red')
        return

    frame_files = glob.glob(os.path.join(frames_dir, 'frame_*.png'))
    if not frame_files:
        cprint(f"Error: No .png frames found in '{frames_dir}'. Nothing to compile.", 'red')
        return

    frame_files.sort(key=get_frame_number_for_sort)
    cprint(f"Found {len(frame_files)} frames to compile with framerate {framerate} fps.", 'yellow')

    list_filename = os.path.join(frames_dir, "ffmpeg_file_list.txt")
    frame_duration = 1.0 / framerate

    try:
        # --- НОВОЕ: Добавляем директиву duration ---
        with open(list_filename, 'w', encoding='utf-8') as f:
            for filename in frame_files:
                safe_path = os.path.abspath(filename).replace('\\', '/')
                f.write(f"file '{safe_path}'\n")
                f.write(f"duration {frame_duration}\n")

        # FFMPEG может потребовать, чтобы последний кадр не имел duration,
        # но современные версии обычно справляются. Этот формат самый надежный.

        # --- Собираем ФИНАЛЬНУЮ, САМУЮ НАДЕЖНУЮ команду FFMPEG ---
        ffmpeg_command = [
            'ffmpeg',
            '-y',                   # Перезаписывать без вопроса
            '-f', 'concat',         # Используем concat demuxer
            '-safe', '0',           # Разрешить абсолютные пути
            '-i', list_filename,    # Входной файл - наш список

            # Настройки кодека остаются теми же
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-profile:v', 'high',
            '-crf', '23',
            '-coder', '1',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',

            # Явно указываем частоту кадров ВЫХОДНОГО файла
            '-r', str(framerate),

            output_filename
        ]

        cprint(f"Running FFMPEG with explicit frame durations...", 'yellow')
        print(f"Command: {' '.join(ffmpeg_command)}")
        print("--- FFMPEG LOG START ---")

        process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='ignore' # Игнорируем ошибки кодировки, если ffmpeg выводит "мусор"
        )

        for line in iter(process.stdout.readline, ''):
            print(line.strip())

        process.stdout.close()
        return_code = process.wait()
        print("--- FFMPEG LOG END ---")

        if return_code != 0:
            cprint(f"\nFFMPEG process failed with return code {return_code}.", 'red')
            cprint("Temporary frames were NOT deleted.", 'cyan')
            return

        cprint(f"\nVideo compilation successful! Output saved to '{os.path.abspath(output_filename)}'", 'green')

        if not keep_frames and os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            cprint(f"Cleaning up frames directory '{frames_dir}'...", 'yellow')
            shutil.rmtree(frames_dir)

    except Exception as e:
        cprint(f"\nAn unexpected error occurred: {e}", 'red')
    finally:
        if os.path.exists(list_filename):
            os.remove(list_filename)


if __name__ == "__main__":
    # ... (Этот блок остается без изменений)
    parser = argparse.ArgumentParser(description="Compile a YouTube-ready video from Project Genesis frames.")
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('-fr', '--framerate', type=int, default=30, help="Framerate of the output video.")
    parser.add_argument('--keep-frames', action='store_true', help="Do not delete the PNG frames folder.")
    args = parser.parse_args()

    RUN_DIR = args.run_directory
    FRAMES_DIR = os.path.join(RUN_DIR, 'frames')
    METADATA_PATH = os.path.join(RUN_DIR, 'metadata.json')

    if not os.path.exists(METADATA_PATH):
        cprint(f"Error: metadata.json not found in '{RUN_DIR}'.", 'red'); exit()
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    cprint(f"\n--- STAGE 3: COMPILING VIDEO for run '{metadata['run_name']}' ---", 'cyan')

    output_filename = os.path.join(RUN_DIR, f"genesis_{metadata['run_name']}_v10.5.mp4")
    keep_frames_flag = args.keep_frames

    compile_video_worker(FRAMES_DIR, output_filename, args.framerate, keep_frames_flag)

    cprint("--- Compilation Complete ---", 'cyan')
