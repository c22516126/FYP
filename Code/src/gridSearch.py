# cqtExperiment.py
# Runs the full transcription pipeline with a configurable filter_scale (Q factor)
# by rebuilding the Basic Pitch model architecture from model2.py and loading
# pretrained weights from the SavedModel.
#
# Place in: Code/src/
#
# Usage:
#   from cqtExperiment import CQTExperimentTranscriber
#   t = CQTExperimentTranscriber(filter_scale=0.5)
#   midi_out, audio_out = t.transcribe("path/to/audio.mp3")

import os
import numpy as np
import tensorflow as tf
import librosa

from pipeline.model2 import model as buildModel
from pipeline.noteCreation import createNotes
from pipeline.generateMIDI import buildMIDI
from pipeline.audioRender import midi_to_audio
from pipeline.stitch import unwrapOutput
from cqtExperiment import loadCQTModel, SAVED_MODEL_DIR
from config import SOUNDFONT_PATH, OUTPUT_DIR, FFT_HOP, WINDOW_SAMPLES, OVERLAP_FRAMES
from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE


from pathlib import Path

class gridSearcher:
    def __init__(
        self,
        filter_scale: float = 1.0,
        output_dir=OUTPUT_DIR,
        soundfont=SOUNDFONT_PATH,
    ):
        self.filter_scale = filter_scale
        self.model = loadCQTModel(filter_scale)

        self.overlap_len = OVERLAP_FRAMES * FFT_HOP
        self.hop_size = WINDOW_SAMPLES - self.overlap_len
        self.FRAMES_PER_WINDOW = WINDOW_SAMPLES // FFT_HOP
        self.FRAMES_PER_STRIDE = self.hop_size // FFT_HOP



        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # tag output files by filter_scale so runs don't overwrite each other
        tag = f"fs{str(filter_scale).replace('.', '_')}"
        self.midi_out  = os.path.join(self.output_dir, f"output_{tag}.mid")
        self.audio_out = os.path.join(self.output_dir, f"output_{tag}.wav")
        self.soundfont = str(soundfont)

    def getNoteList(self, audio_path: str):
        pitchFull, onsetFull = self._run_inference(audio_path)

        notes = createNotes(
            pitchFull,
            onsetPost=onsetFull,
            sampleRate=22050,
            fftHop=FFT_HOP,
        )


        return self.midi_out, self.audio_out

    def _run_inference(self, audio_path: str):
        # load audio the same way BP does
        audio, _ = librosa.load(str(audio_path), sr=22050, mono=True)
        audio = audio.astype(np.float32)

        pitchWindows = []
        onsetWindows = []

        start = 0
        while start < len(audio):
            end   = start + AUDIO_N_SAMPLES
            chunk = audio[start:start + WINDOW_SAMPLES]
            if len(chunk) < WINDOW_SAMPLES:
                chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
            # pad to model's expected input size
            chunk = np.pad(chunk, (0, AUDIO_N_SAMPLES - WINDOW_SAMPLES))

            # model expects (batch, samples, channels) — same shape as get_audio_input yields
            inp = tf.constant(chunk[np.newaxis, :, np.newaxis], dtype=tf.float32)
            out = self.model(inp, training=False)

            pitchWindows.append(out["note"].numpy().squeeze(0))
            onsetWindows.append(out["onset"].numpy().squeeze(0))

            start += self.hop_size

        pitchStack = np.stack(pitchWindows)
        onsetStack = np.stack(onsetWindows)

        pitchFull = unwrapOutput(pitchStack, self.FRAMES_PER_WINDOW, self.FRAMES_PER_STRIDE)
        onsetFull = unwrapOutput(onsetStack, self.FRAMES_PER_WINDOW, self.FRAMES_PER_STRIDE)

        return pitchFull, onsetFull
