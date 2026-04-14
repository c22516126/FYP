import os
import numpy as np

from src.pipeline.loadModel import loadModel
from src.pipeline.inference import infer
from src.pipeline.noteCreation import createNotes, framesToSeconds
from src.pipeline.generateMIDI import buildMIDI
from src.pipeline.stitch import unwrapOutput
from src.config import MODEL_PATH, OUTPUT_DIR, FFT_HOP, WINDOW_SAMPLES, OVERLAP_FRAMES, AUDIO_SAMPLE_RATE, ONSET_DEFAULT, FRAME_DEFAULT, ENERGY_DEFAULT, MIN_DEFAULT

from basic_pitch.inference import get_audio_input

class Transcriber:
    def __init__(
        self,
        model_path = MODEL_PATH,
        output_dir = OUTPUT_DIR,
    ):
        
        self.interpreter = loadModel(model_path)

        self.overlap_len = OVERLAP_FRAMES * FFT_HOP
        self.hop_size = WINDOW_SAMPLES - self.overlap_len

        self.FRAMES_PER_WINDOW = WINDOW_SAMPLES // FFT_HOP
        self.FRAMES_PER_STRIDE = self.hop_size // FFT_HOP

        assert WINDOW_SAMPLES % FFT_HOP == 0, \
            "Window size must be divisible by model hop"
        assert self.hop_size % FFT_HOP == 0, \
            "Hop size must be divisible by model hop"
        
        # outputs
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok = True)

        self.midi_out = os.path.join(self.output_dir, "output.mid")

    def transcribe(self, audio_path):
        
        # run inference
        pitchFull, onsetFull, audio_len = self._run_inference(audio_path)

        notes = createNotes(
            frames=pitchFull,
            onsets=onsetFull,
            onsetThreshold=0.7,
            frameThreshold=0.3,
            minimumNoteLength=11,
            energyTolerance=ENERGY_DEFAULT,
            melodia=False
        )

        notesInSeconds = framesToSeconds(
            notes,
            sampleRate=AUDIO_SAMPLE_RATE,
            hopSize=FFT_HOP
        )

        # create midi
        print(type(notesInSeconds[0]))
        buildMIDI(notesInSeconds, self.midi_out)

        return self.midi_out


    def _run_inference(self, audio_path):
        pitchWindows = []
        onsetWindows = []
        audioOriginalLen = None

        for window, _, origLen in get_audio_input(
            audio_path,
            overlap_len=self.overlap_len,
            hop_size=self.hop_size,
        ):
            audioOriginalLen = origLen
            p, o, _ = infer(self.interpreter, window)
            pitchWindows.append(p)
            onsetWindows.append(o)

        pitchStack = np.stack(pitchWindows)
        onsetStack = np.stack(onsetWindows)

        pitchFull = unwrapOutput(
        pitchStack,
        self.FRAMES_PER_WINDOW,
        self.FRAMES_PER_STRIDE,
    )

        onsetFull = unwrapOutput(
            onsetStack,
            self.FRAMES_PER_WINDOW,
            self.FRAMES_PER_STRIDE,
        )

        return pitchFull, onsetFull, audioOriginalLen
    
# get pitch and interval arrays using tunable note creation params
def transcribeWithParams(pitchFull, onsetFull, params=None):
    if params is None:
        params = {
            "onset": 0.5,
            "frame": 0.3,
            "min_len": 11,
            "energy": 8,
            "melodia": True
        }
    notes = createNotes(
        frames=pitchFull,
        onsets=onsetFull,
        onsetThreshold=params["onset"],
        frameThreshold=params["frame"],
        minimumNoteLength=params["min_len"],
        energyTolerance=params["energy"],
        melodia=params.get("melodia", False)
    )

    return notes
