import os
import sys
from pathlib import Path

# Add src directory to Python path for all tests
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))