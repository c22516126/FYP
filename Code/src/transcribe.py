import os
import numpy as np

from pipeline.loadModel import loadModel
from pipeline.inference import infer
from pipeline.noteCreation import createNotes
from pipeline.generateMIDI import buildMIDI
from pipeline.audioRender import midi_to_audio
from pipeline.stitch import unwrapOutput
from config import MODEL_PATH, SOUNDFONT_PATH, OUTPUT_DIR, FFT_HOP, WINDOW_SAMPLES, OVERLAP_FRAMES

from basic_pitch.inference import get_audio_input

class Transcriber:
    def __init__(
        self,
        model_path = MODEL_PATH,
        output_dir = OUTPUT_DIR,
        soundfont = SOUNDFONT_PATH
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
        self.audio_out = os.path.join(self.output_dir, "output.wav")

        self.soundfont = str(soundfont)

    def transcribe(self, audio_path):
        
        # run inference
        pitchFull, onsetFull, audio_len = self._run_inference(audio_path)

        # create notes
        notes = createNotes(
            pitchFull,
            onsetPost=onsetFull,
            sampleRate=22050,
            fftHop= FFT_HOP,
        )

        # create midi
        buildMIDI(notes, self.midi_out)

        # render audio
        midi_to_audio(self.midi_out, self.audio_out, self.soundfont)

        return self.midi_out, self.audio_out


    # OPTIMIZE THIS LATER
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
