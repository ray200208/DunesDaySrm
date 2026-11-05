import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
src_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from preprocess_etm import run_aggregation

if __name__ == "__main__":
    run_aggregation()
