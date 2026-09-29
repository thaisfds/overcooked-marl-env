#!/usr/bin/env python3
"""
Legacy alias for visualize_model.py.
Redirects execution to visualize_model.py for backward compatibility.
"""
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from visualize_model import main

if __name__ == "__main__":
    main()
