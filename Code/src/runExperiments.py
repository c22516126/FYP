import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cqtExperiment import CQTExperimentTranscriber

AUDIO_PATH = r"C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\CDL\cdlEVAL.mp3"

# start with baseline to verify weights load correctly
t = CQTExperimentTranscriber(filter_scale=0.75)
midi, audio = t.transcribe(AUDIO_PATH)
print(f"MIDI: {midi}")
print(f"Audio: {audio}")