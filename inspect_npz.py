# inspect_npz.py
# A simple utility to inspect the contents of a .npz file.

import numpy as np
import argparse
from termcolor import cprint
import os

def inspect_file(file_path):
    """Loads a .npz file and prints information about its contents."""
    if not os.path.exists(file_path):
        cprint(f"Error: File not found at '{file_path}'", 'red')
        return

    cprint(f"\n--- Inspecting file: {os.path.basename(file_path)} ---", 'yellow')

    try:
        with np.load(file_path, allow_pickle=True) as data:
            cprint("Archive successfully loaded.", 'green')

            # .files - это атрибут, который содержит список всех ключей (имен массивов) в архиве.
            keys = data.files

            if not keys:
                cprint("The archive is empty (contains no arrays).", 'yellow')
                return

            cprint(f"Found {len(keys)} array(s) with the following keys:", 'cyan')
            for key in keys:
                array = data[key]
                # Получаем информацию о массиве
                shape = array.shape
                dtype = array.dtype

                # Специальная обработка для скалярных массивов, созданных через np.array(...)
                if shape == ():
                    shape_str = "scalar"
                    value_preview = f"value={array.item()}"
                else:
                    shape_str = str(shape)
                    value_preview = "" # Не будем выводить большие массивы

                print(f"  - Key: '{key}' | Shape: {shape_str} | DType: {dtype} {value_preview}")

    except Exception as e:
        cprint(f"An error occurred while trying to read the file: {e}", 'red')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect the contents of a single .npz data frame from a Project Genesis run."
    )
    parser.add_argument('run_directory', type=str, help="Path to the run directory.")
    parser.add_argument('frame_number', type=int, help="The specific frame number to inspect.")
    args = parser.parse_args()

    data_file_path = os.path.join(
        args.run_directory,
        'data',
        f"frame_{args.frame_number:05d}.npz"
    )

    inspect_file(data_file_path)
