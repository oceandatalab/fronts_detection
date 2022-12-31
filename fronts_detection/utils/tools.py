import os
import sys


def load_python_file(file_path: str):
    """Load a file and parse it as a Python module."""
    if not os.path.exists(file_path):
        _msg = f'File not found: {file_path}'
        raise IOError(_msg)

    full_path = os.path.abspath(file_path)
    python_filename = os.path.basename(full_path)
    module_name, _ = os.path.splitext(python_filename)
    module_dir = os.path.dirname(full_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    module = __import__(module_name, globals(), locals(), [], 0)
    return module
