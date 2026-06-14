"""Launch Jetson ANC pipeline. Run: python run_jetson.py [args]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hardware.crn_pipeline import main
if __name__ == "__main__":
    main()
