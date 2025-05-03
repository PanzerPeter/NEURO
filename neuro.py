#!/usr/bin/env python
import sys
import os

# Add the 'src' directory to the Python path
# This allows importing modules from 'src' when running neuro.py from the root
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from src.neuro import main
except ImportError as e:
    print(f"Error importing main function: {e}", file=sys.stderr)
    print("Please ensure 'src' directory is in the Python path or run using 'python -m src.neuro'", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
