# compiler.py v1.0
"""
Project Genesis - Video Compilation Module
------------------------------------------
- Собирает PNG-кадры в видеофайл MP4 с помощью ffmpeg.
- Обрабатывает ошибки и удаляет временные файлы.
"""
import subprocess
import os
import shutil
from termcolor import cprint

def compile_video(frames_dir, output_filename, framerate=25, keep_frames=False):
    """Собирает видео из кадров."""
    print("\n--- STAGE 3: COMPILING VIDEO ---")
    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        cprint(f"Error: Frames directory '{frames_dir}' is empty or does not exist.", 'red')
        return

    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', os.path.join(frames_dir, 'frame_%05d.png'),
        '-c:v', 'libx264',      # Хороший кодек
        '-pix_fmt', 'yuv420p',  # Для совместимости с большинством плееров
        '-y',                   # Перезаписывать файл без вопроса
        output_filename
    ]

    try:
        print(f"Running FFMPEG to create '{output_filename}'...")
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        cprint(f"Video compilation successful!", 'green')
    except subprocess.CalledProcessError as e:
        cprint("\n--- FFMPEG ERROR ---", 'red')
        print(e.stderr)
        cprint("--------------------", 'red')
    except FileNotFoundError:
        cprint("\nError: `ffmpeg` command not found.", 'red')
        cprint("Please install ffmpeg and ensure it is in your system's PATH.", 'yellow')

    finally:
        if not keep_frames and os.path.exists(frames_dir):
            print(f"Cleaning up temporary frames directory '{frames_dir}'...")
            shutil.rmtree(frames_dir)
