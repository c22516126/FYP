from pathlib import Path

# Root of the project (src parent)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Model path (relative to installed basic_pitch)
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

# SoundFont
SOUNDFONT_PATH = PROJECT_ROOT / "resources" / "FluidR3_GM.sf2"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
