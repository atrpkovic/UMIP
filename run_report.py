import os
import sys

# Allow running without install: add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from umip.main import run

if __name__ == "__main__":
    run()
