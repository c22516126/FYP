from loadModel import loadTfliteModel
from inference import runModelFast, unwrapOutput
from noteCreation import createNotes
from generateMIDI import buildMIDI
from audioRender import midi_to_audio

from basic_pitch.inference import get_audio_input

import numpy as np

AUDIO_FILE = "testInput/ClairDeLune.mp3"
MODEL_PATH = (
    "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/code/venv/"
    "Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"
)

print("Loading model...")
interpreter = loadTfliteModel(MODEL_PATH)
runner = interpreter.get_signature_runner()

# BasicPitch settings
FFT_HOP = 512
AUDIO_N_SAMPLES = 32768
DEFAULT_OVERLAPPING_FRAMES = 30

nOverlap = DEFAULT_OVERLAPPING_FRAMES
overlapLen = nOverlap * FFT_HOP
hopSize = AUDIO_N_SAMPLES - overlapLen

pitchWindows = []
onsetWindows = []
offsetWindows = []

audioOriginalLen = None

print("Running fast inference...")

for window, _, origLen in get_audio_input(AUDIO_FILE, overlap_len=overlapLen, hop_size=hopSize):
    audioOriginalLen = origLen
    p, o, c = runModelFast(interpreter, window)
    pitchWindows.append(p)
    onsetWindows.append(o)
    offsetWindows.append(c)

# Convert lists -> (B, F, 88)
pitchStack = np.stack(pitchWindows)
onsetStack = np.stack(onsetWindows)
offsetStack = np.stack(offsetWindows)

# Unwrap like BasicPitch
pitchFull = unwrapOutput(pitchStack, audioOriginalLen, nOverlap, hopSize)
onsetFull = unwrapOutput(onsetStack, audioOriginalLen, nOverlap, hopSize)
offsetFull = unwrapOutput(offsetStack, audioOriginalLen, nOverlap, hopSize)

# Extract notes
notes = createNotes(
    pitchFull,
    onsetPost=onsetFull,
    sr=22050,
    fft_hop=512,
)

print("Detected notes:", len(notes))

# Build MIDI
buildMIDI(notes, "demo_output.mid")
print("Saved demo_output.mid")
midi_to_audio(
        midi_path="demo_output.mid",
        wav_path="output.wav",
        soundfont="FluidR3_GM.sf2"
    )
