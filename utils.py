# utils.py v1.0
# A central module for common, robust utility functions.

import numpy as np

def is_valid_array(arr):
    """
    The one and only function to safely check if an object is a non-empty
    numpy array or a non-empty list.

    This function is designed to be used in `if` statements to avoid the
    `ValueError: The truth value of an array... is ambiguous` error.

    Returns:
        bool: True if the object is a list/array with at least one element.
    """

    # Check 1: Is it a numpy array?
    if isinstance(arr, np.ndarray):
        # For numpy arrays, the correct way to check for non-emptiness is .size
        return arr.size > 0

    # Check 2: Is it a list? (or tuple, etc.)
    if isinstance(arr, (list, tuple)):
        # For standard Python sequences, `len()` is correct.
        return len(arr) > 0

    # If it's something else (None, an int, etc.), it's not a valid array/list.
    return False
