import time
import numpy as np

from loadModel import loadTfliteModel
from inference import runModelFast, unwrapOutput
from noteCreation import createNotes
from generateMIDI import buildMIDI
from basic_pitch.inference import get_audio_input

# ==== BasicPitch constants (drop-in stable versions) ====
FFT_HOP = 512
AUDIO_N_SAMPLES = 32768
DEFAULT_OVERLAPPING_FRAMES = 30
# ========================================================

AUDIO_FILE = "testInput/Avril14th.mp3"
MODEL_PATH = (
    "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/code/venv/"
    "Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"
)


# ==========================================================
# TEST 1 — Fast Inference Per Window
# ==========================================================

def test_inference_speed():
    print("\n=== TEST 1: Fast Window Inference Speed ===")

    interpreter = loadTfliteModel(MODEL_PATH)

    window_times = []
    count = 0

    for window, _, _ in get_audio_input(AUDIO_FILE, overlap_len=30, hop_size=512):
        start = time.time()
        runModelFast(interpreter, window)
        end = time.time()

        window_times.append(end - start)
        count += 1

    print(f"Windows processed: {count}")
    print(f"Avg fast inference per window: {np.mean(window_times):.5f} sec")
    print(f"Max time: {np.max(window_times):.5f} sec")
    print(f"Min time: {np.min(window_times):.5f} sec")


# ==========================================================
# TEST 2 — Stitching + Unwrap Correctness
# ==========================================================

def test_stitch_shapes():
    print("\n=== TEST 2: Stitching + Unwrapped Shape Check ===")

    interpreter = loadTfliteModel(MODEL_PATH)

    pitch_windows = []
    onset_windows = []
    offset_windows = []

    audio_orig_len = None

    # Collect windows
    for window, _, orig_len in get_audio_input(AUDIO_FILE, overlap_len=30, hop_size=512):
        audio_orig_len = orig_len
        p, o, c = runModelFast(interpreter, window)

        pitch_windows.append(p)
        onset_windows.append(o)
        offset_windows.append(c)

    # Convert to Batched (B, F, 88)
    pitch_stack = np.stack(pitch_windows)
    onset_stack = np.stack(onset_windows)
    offset_stack = np.stack(offset_windows)

    # Unwrap like BasicPitch
    nOverlap = DEFAULT_OVERLAPPING_FRAMES
    overlapLen = nOverlap * FFT_HOP
    hopSize = AUDIO_N_SAMPLES - overlapLen

    pitchFull = unwrapOutput(pitch_stack, audio_orig_len, nOverlap, hopSize)
    onsetFull = unwrapOutput(onset_stack, audio_orig_len, nOverlap, hopSize)
    offsetFull = unwrapOutput(offset_stack, audio_orig_len, nOverlap, hopSize)

    print("Pitch full shape :", pitchFull.shape)
    print("Onset full shape :", onsetFull.shape)
    print("Offset full shape:", offsetFull.shape)

    assert pitchFull.ndim == 2, "Pitch must be 2D"
    assert pitchFull.shape[1] == 88, "Pitch must have 88 bins"


# ==========================================================
# TEST 3 — FULL PIPELINE PERFORMANCE
# ==========================================================

def test_full_pipeline():
    print("\n=== TEST 3: Full Fast Pipeline Timing ===")

    from inference import runModelFast, unwrapOutput
from basic_pitch.constants import FFT_HOP, AUDIO_N_SAMPLES, DEFAULT_OVERLAPPING_FRAMES

def test_full_fast_pipeline():
    print("\n=== TEST 3: Full Fast Pipeline Timing ===")

    interpreter = loadTfliteModel(MODEL_PATH)

    # BasicPitch settings
    nOverlap = DEFAULT_OVERLAPPING_FRAMES
    overlapLen = nOverlap * FFT_HOP
    hopSize = AUDIO_N_SAMPLES - overlapLen

    pitchWindows = []
    onsetWindows = []
    offsetWindows = []

    t0 = time.time()

    for window, _, origLen in get_audio_input(AUDIO_FILE, overlap_len=overlapLen, hop_size=hopSize):
        p, o, c = runModelFast(interpreter, window)
        pitchWindows.append(p)
        onsetWindows.append(o)
        offsetWindows.append(c)
    
    t1 = time.time()

    pitchFull = unwrapOutput(np.stack(pitchWindows), origLen, nOverlap, hopSize)

    t2 = time.time()

    notes = createNotes(pitchFull)
    buildMIDI(notes, "test_output.mid")

    t3 = time.time()

    print("Inference time      :", t1 - t0)
    print("Note extraction time:", t2 - t1)
    print("MIDI build time     :", t3 - t2)
    print("TOTAL pipeline      :", t3 - t0)
    print("Notes:", len(notes))



# ==========================================================

if __name__ == "__main__":
    test_inference_speed()
    test_stitch_shapes()
    test_full_pipeline()
