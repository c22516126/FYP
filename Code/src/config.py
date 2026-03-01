# paths
# ----------------------------------------------------------------------------------------------------------------------------------------
from pathlib import Path

# project root
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# relative model path
MODEL_PATH = (
    PROJECT_ROOT
    / "venv"
    / "Lib"
    / "site-packages"
    / "basic_pitch"
    / "saved_models"
    / "icassp_2022"
    / "nmp.tflite"
)

# soundfont
SOUNDFONT_PATH = PROJECT_ROOT / "resources" / "FluidR3_GM.sf2"

# output location
OUTPUT_DIR = PROJECT_ROOT / "output"


# variables
# ----------------------------------------------------------------------------------------------------------------------------------------

FFT_HOP = 256 # for every 256 samples, start a frame and make a prediction
WINDOW_SAMPLES = 32768 # amount of samples fed to model during each inference instance
OVERLAP_FRAMES = 30 # amount of frames overlapping within a window