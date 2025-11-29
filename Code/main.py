from loadModel import loadTfliteModel
from inference import runModel
from noteCreation import createNotes
from generateMIDI import buildMIDI
import numpy as np
from basic_pitch.inference import get_audio_input

AUDIO_FILE = "testInput/Avril14th.mp3"
MODEL_PATH = "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/code/venv/Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"

interpreter = loadTfliteModel(MODEL_PATH)

allPitch = []
allOnset = []
allOffset = []

# 1. Inference across ALL windows
for audioWindow, _, _ in get_audio_input(AUDIO_FILE, overlap_len=30, hop_size=512):
    pitch, onset, offset = runModel(interpreter, audioWindow)
    allPitch.append(pitch)
    allOnset.append(onset)
    allOffset.append(offset)

# 2. Stitch the full posteriorgrams
pitchPost = np.concatenate(allPitch, axis=0)
onsetPost = np.concatenate(allOnset, axis=0)
offsetPost = np.concatenate(allOffset, axis=0)

# 3. Extract notes from the FULL posteriorgram (correct)
notes = createNotes(pitchPost)

# 4. Build ONE final MIDI file (correct)
print("Number of notes:", len(notes))   # ← PUT IT HERE
buildMIDI(notes, "demo_output.mid")
