import os
import numpy as np

from pipeline.loadModel import loadTfliteModel
from pipeline.inference import runModelFast, unwrapOutput
from pipeline.noteCreation import createNotes
from pipeline.generateMIDI import buildMIDI
from pipeline.audioRender import midi_to_audio
from config import MODEL_PATH, SOUNDFONT_PATH, OUTPUT_DIR

from basic_pitch.inference import get_audio_input


class TranscriptionCLI:
    """
    End-to-end pipeline wrapper.
    Only user input: audio file path.
    Output always written to /output.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        soundfont=SOUNDFONT_PATH,
        
    ):
        print("Loading model...")
        self.interpreter = loadTfliteModel(model_path)
        self.runner = self.interpreter.get_signature_runner()

        # -----------------
        # Audio + model config
        # -----------------
        self.FFT_HOP = 512
        self.MODEL_HOP = 512  # samples per model frame (fixed by model)

        self.AUDIO_N_SAMPLES = 32768
        self.N_OVERLAP_FRAMES = 30

        # overlap + stride in samples
        self.overlap_len = self.N_OVERLAP_FRAMES * self.FFT_HOP
        self.hop_size = self.AUDIO_N_SAMPLES - self.overlap_len

        # -----------------
        # Frame-domain quantities (THIS was the missing ordering)
        # -----------------
        self.FRAMES_PER_WINDOW = self.AUDIO_N_SAMPLES // self.MODEL_HOP
        self.FRAMES_PER_STRIDE = self.hop_size // self.MODEL_HOP

        # safety checks (keep these)
        assert self.AUDIO_N_SAMPLES % self.MODEL_HOP == 0, \
            "Window size must be divisible by model hop"
        assert self.hop_size % self.MODEL_HOP == 0, \
            "Hop size must be divisible by model hop"

        # -----------------
        # Output paths
        # -----------------
        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.midi_out = os.path.join(self.output_dir, "output.mid")
        self.audio_out = os.path.join(self.output_dir, "output.wav")
        self.soundfont = str(soundfont)


    def transcribe(self, audio_path):
        print("\n=== STEP 1: Running inference ===")
        pitchFull, onsetFull, audio_len = self._run_inference(audio_path)

        print("=== STEP 2: Extracting notes ===")
        notes = createNotes(
            pitchFull,
            onsetPost=onsetFull,
            sr=22050,
            fft_hop=self.FFT_HOP,
        )
        print(f"Detected {len(notes)} notes")

        print("=== STEP 3: Building MIDI ===")
        buildMIDI(notes, self.midi_out)

        print("=== STEP 4: Rendering audio ===")
        midi_to_audio(self.midi_out, self.audio_out, self.soundfont)

        print(f"\nDone!\nMIDI  → {self.midi_out}\nAudio → {self.audio_out}")
        return self.midi_out, self.audio_out

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
            p, o, _ = runModelFast(self.interpreter, window)
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
