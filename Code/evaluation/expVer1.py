"""
CQT Filter Scale Experiment
----------------------------
Rebuilds the Basic Pitch model architecture with different filter_scale (Q) values,
visualises the resulting CQT spectrograms, and scores each configuration against
the Clair de Lune reference MIDI using mir_eval.

Filter scale controls Q:
    Q = filter_scale / (2^(1/bins_per_octave) - 1)

    filter_scale=0.5  -> lower Q  -> wider filters, better time res, worse freq res
    filter_scale=1.0  -> default  -> Basic Pitch baseline
    filter_scale=2.0  -> higher Q -> narrower filters, better freq res, worse time res

Usage:
    python cqt_experiment.py
"""

import sys
import os
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pretty_midi
import mir_eval

# ── path setup ────────────────────────────────────────────────────────────────
# Adjust this so Python can find your src/ modules
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ── Basic Pitch imports ────────────────────────────────────────────────────────
from basic_pitch.layers import nnaudio, signal
from basic_pitch import nn
from basic_pitch.constants import (
    ANNOTATIONS_BASE_FREQUENCY,   # 27.5 Hz  (A0, lowest piano key)
    ANNOTATIONS_N_SEMITONES,       # 88
    AUDIO_N_SAMPLES,               # 44032
    AUDIO_SAMPLE_RATE,             # 22050
    CONTOURS_BINS_PER_SEMITONE,    # 3
    FFT_HOP,                       # 256
    N_FREQ_BINS_CONTOURS,          # 264
)

# ── your pipeline imports ──────────────────────────────────────────────────────
from src.pipeline.noteCreation import createNotes
from src.pipeline.generateMIDI import buildMIDI
from src.pipeline.stitch import unwrapOutput
from src.config import (
    OVERLAP_FRAMES,
    WINDOW_SAMPLES,
    CDL_REFERENCE_PATH,
    SOUNDFONT_PATH,
)

# ── experiment config ──────────────────────────────────────────────────────────
AUDIO_PATH      = r"C:\Users\jason\school\FYP\FYP\Code\input\ClairDeLune.mp3"
SAVED_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "venv", "Lib", "site-packages",
    "basic_pitch", "saved_models", "icassp_2022", "nmp"   # saved_model.pb lives here
)
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "cqt_experiment_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILTER_SCALES   = [0.5, 1.0, 2.0]   # Q experiment values
LABELS          = ["Low Q (0.5)", "Baseline (1.0)", "High Q (2.0)"]
COLOURS         = ["tab:blue", "tab:orange", "tab:green"]

# note creation thresholds — same as your main pipeline
PITCH_THRESHOLD = 0.2
ONSET_THRESHOLD = 0.5
MIN_FRAMES      = 3
BRIDGE_GAP      = 2


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))

def build_model_with_filter_scale(filter_scale: float) -> tf.keras.Model:
    """
    Rebuild the Basic Pitch architecture with a custom filter_scale (Q factor).
    Weights are loaded from the pretrained SavedModel so only the CQT
    preprocessing changes — the CNN weights stay identical.
    """
    n_harmonics = 8
    n_semitones = np.min([
        int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
        MAX_N_SEMITONES,
    ])

    tfkl = tf.keras.layers

    def _initializer():
        return tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg", distribution="uniform", seed=None
        )

    def _kernel_constraint():
        return tf.keras.constraints.UnitNorm(axis=[0, 1, 2])

    # ── input + CQT (filter_scale is injected here) ──────────────────────────
    inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))
    x = nn.FlattenAudioCh()(inputs)
    x = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
        filter_scale=filter_scale,          # <── this is the only change
    )(x)
    x = signal.NormalizedLog()(x)
    x = tf.expand_dims(x, -1)
    x = tfkl.BatchNormalization()(x)

    # ── harmonic stacking ─────────────────────────────────────────────────────
    x = nn.HarmonicStacking(
        CONTOURS_BINS_PER_SEMITONE,
        [0.5] + list(range(1, n_harmonics)),
        N_FREQ_BINS_CONTOURS,
    )(x)

    # ── contour branch ────────────────────────────────────────────────────────
    x_contours = tfkl.Conv2D(32, (5, 5), padding="same",
                             kernel_initializer=_initializer(),
                             kernel_constraint=_kernel_constraint())(x)
    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)

    x_contours = tfkl.Conv2D(8, (3, 3 * 13), padding="same",
                             kernel_initializer=_initializer(),
                             kernel_constraint=_kernel_constraint())(x)
    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)

    x_contours = tfkl.Conv2D(1, (5, 5), padding="same", activation="sigmoid",
                             kernel_initializer=_initializer(),
                             kernel_constraint=_kernel_constraint(),
                             name="contours-reduced")(x_contours)
    x_contours = nn.FlattenFreqCh(name="contour")(x_contours)
    x_contours_reduced = tf.expand_dims(x_contours, -1)

    x_contours_reduced = tfkl.Conv2D(32, (7, 7), padding="same", strides=(1, 3),
                                     kernel_initializer=_initializer(),
                                     kernel_constraint=_kernel_constraint())(x_contours_reduced)
    x_contours_reduced = tfkl.ReLU()(x_contours_reduced)

    # ── note branch ───────────────────────────────────────────────────────────
    x_notes_pre = tfkl.Conv2D(1, (7, 3), padding="same", activation="sigmoid",
                              kernel_initializer=_initializer(),
                              kernel_constraint=_kernel_constraint())(x_contours_reduced)
    x_notes = nn.FlattenFreqCh(name="note")(x_notes_pre)

    # ── onset branch ─────────────────────────────────────────────────────────
    x_onset = tfkl.Conv2D(32, (5, 5), padding="same", strides=(1, 3),
                          kernel_initializer=_initializer(),
                          kernel_constraint=_kernel_constraint())(x)
    x_onset = tfkl.BatchNormalization()(x_onset)
    x_onset = tfkl.ReLU()(x_onset)
    x_onset = tfkl.Concatenate(axis=3, name="concat")([x_notes_pre, x_onset])
    x_onset = tfkl.Conv2D(1, (3, 3), padding="same", activation="sigmoid",
                          kernel_initializer=_initializer(),
                          kernel_constraint=_kernel_constraint())(x_onset)
    x_onset = nn.FlattenFreqCh(name="onset")(x_onset)

    model = tf.keras.Model(inputs=inputs, outputs={
        "onset": x_onset, "contour": x_contours, "note": x_notes
    })

    # ── load pretrained weights ───────────────────────────────────────────────
    pretrained = tf.saved_model.load(SAVED_MODEL_DIR)
    pretrained_vars = {v.name: v for v in pretrained.variables}
    loaded, skipped = 0, 0
    for var in model.variables:
        if var.name in pretrained_vars:
            var.assign(pretrained_vars[var.name])
            loaded += 1
        else:
            skipped += 1
    print(f"  Weights loaded: {loaded}, skipped (CQT layer, expected): {skipped}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2.  AUDIO LOADING + WINDOWING
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=AUDIO_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def window_audio(audio: np.ndarray):
    """Yield overlapping windows matching your Transcriber logic."""
    overlap_len  = OVERLAP_FRAMES * FFT_HOP
    hop_size     = WINDOW_SAMPLES - overlap_len
    frames_per_window = WINDOW_SAMPLES // FFT_HOP
    frames_per_stride = hop_size     // FFT_HOP

    windows = []
    start = 0
    while start < len(audio):
        end    = start + WINDOW_SAMPLES
        chunk  = audio[start:end]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk)
        start += hop_size

    return windows, frames_per_window, frames_per_stride


# ══════════════════════════════════════════════════════════════════════════════
# 3.  INFERENCE WITH A KERAS MODEL
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model: tf.keras.Model, audio: np.ndarray):
    windows, fpw, fps = window_audio(audio)
    pitch_windows, onset_windows = [], []

    for w in windows:
        inp = tf.constant(w[np.newaxis, :, np.newaxis], dtype=tf.float32)  # (1, N, 1)
        out = model(inp, training=False)
        pitch_windows.append(out["note"].numpy().squeeze(0))
        onset_windows.append(out["onset"].numpy().squeeze(0))

    pitch_stack = np.stack(pitch_windows)
    onset_stack = np.stack(onset_windows)

    pitch_full = unwrapOutput(pitch_stack, fpw, fps)
    onset_full = unwrapOutput(onset_stack, fpw, fps)
    return pitch_full, onset_full


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CQT SPECTROGRAM (standalone, no inference)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cqt_spectrogram(audio: np.ndarray, filter_scale: float) -> np.ndarray:
    n_harmonics = 8
    n_semitones = np.min([
        int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
        MAX_N_SEMITONES,
    ])
    cqt_layer = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
        filter_scale=filter_scale,
    )
    # use first 5 seconds for visualisation
    clip = audio[:AUDIO_SAMPLE_RATE * 5]
    inp  = tf.constant(clip[np.newaxis, :], dtype=tf.float32)
    spec = cqt_layer(inp).numpy().squeeze(0)   # (time, freq)
    return spec.T   # (freq, time) — conventional spectrogram orientation


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_midi(estimate_path: str, reference_path: str) -> dict:
    def load(path):
        mid  = pretty_midi.PrettyMIDI(path)
        notes = mid.instruments[0].notes
        intervals = np.array([[n.start, n.end] for n in notes])
        pitches   = np.array([mir_eval.util.midi_to_hz(n.pitch) for n in notes])
        return intervals, pitches

    est_i, est_p = load(estimate_path)
    ref_i, ref_p = load(reference_path)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, offset_ratio=None
    )
    return {"precision": p, "recall": r, "f1": f}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading audio...")
    audio = load_audio(AUDIO_PATH)

    # ── A: Spectrogram visualisation ─────────────────────────────────────────
    print("\nComputing CQT spectrograms for visualisation...")
    specs = []
    for fs in FILTER_SCALES:
        print(f"  filter_scale={fs}")
        specs.append(compute_cqt_spectrogram(audio, fs))

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("CQT Spectrogram — Effect of Filter Scale (Q Factor)", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    for i, (spec, label, colour) in enumerate(zip(specs, LABELS, COLOURS)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(spec, aspect="auto", origin="lower",
                       extent=[0, 5, 0, spec.shape[0]], cmap="magma")
        ax.set_title(label, color=colour, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency bin")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # bandwidth comparison plot
    ax_bw = fig.add_subplot(gs[1, :])
    freq_bins = np.arange(1, specs[0].shape[0] + 1)
    for fs, label, colour in zip(FILTER_SCALES, LABELS, COLOURS):
        # bandwidth = fk / Q,  Q = filter_scale / (2^(1/bpo) - 1)
        bpo = 12 * CONTOURS_BINS_PER_SEMITONE
        Q   = fs / (2 ** (1 / bpo) - 1)
        freqs = ANNOTATIONS_BASE_FREQUENCY * (2 ** (np.arange(len(freq_bins)) / bpo))
        bw    = freqs / Q
        ax_bw.plot(freqs, bw, label=f"{label}  (Q={Q:.1f})", color=colour)

    ax_bw.set_xlabel("Center frequency (Hz)")
    ax_bw.set_ylabel("Bandwidth Δf (Hz)")
    ax_bw.set_title("Filter Bandwidth vs Frequency for each Q value")
    ax_bw.legend()
    ax_bw.set_xscale("log")

    spec_path = os.path.join(OUTPUT_DIR, "cqt_spectrogram_comparison.png")
    plt.savefig(spec_path, dpi=150, bbox_inches="tight")
    print(f"\nSpectrogram figure saved → {spec_path}")
    plt.close()

    # ── B: Inference + scoring ────────────────────────────────────────────────
    results = []
    for fs, label in zip(FILTER_SCALES, LABELS):
        print(f"\n{'='*60}")
        print(f"Running inference: {label}")
        print(f"{'='*60}")

        model = build_model_with_filter_scale(fs)

        pitch_full, onset_full = run_inference(model, audio)

        notes = createNotes(
            pitch_full,
            onsetPost=onset_full,
            pitchThreshold=PITCH_THRESHOLD,
            onsetThreshold=ONSET_THRESHOLD,
            minFrames=MIN_FRAMES,
            bridgeGap=BRIDGE_GAP,
            sampleRate=AUDIO_SAMPLE_RATE,
            fftHop=FFT_HOP,
        )

        midi_out = os.path.join(OUTPUT_DIR, f"cdl_fs{str(fs).replace('.', '_')}.mid")
        buildMIDI(notes, midi_out)
        print(f"  MIDI saved → {midi_out}")

        scores = score_midi(midi_out, CDL_REFERENCE_PATH)
        scores["filter_scale"] = fs
        scores["label"]        = label
        scores["n_notes"]      = len(notes)
        results.append(scores)

        print(f"  Precision : {scores['precision']:.4f}")
        print(f"  Recall    : {scores['recall']:.4f}")
        print(f"  F1        : {scores['f1']:.4f}")
        print(f"  Notes     : {scores['n_notes']}")

        tf.keras.backend.clear_session()

    # ── C: Results bar chart ──────────────────────────────────────────────────
    fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle("AMT Performance vs CQT Filter Scale (Clair de Lune)", fontsize=13, fontweight="bold")

    metrics = ["precision", "recall", "f1"]
    titles  = ["Precision", "Recall", "F1 Score"]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [r[metric] for r in results]
        bars = ax.bar(LABELS, vals, color=COLOURS, alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    scores_path = os.path.join(OUTPUT_DIR, "cqt_scores_comparison.png")
    plt.savefig(scores_path, dpi=150, bbox_inches="tight")
    print(f"\nScores figure saved → {scores_path}")
    plt.close()

    # ── D: Print summary table ────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Label':<20} {'Filter Scale':>12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Notes':>7}")
    print("="*60)
    for r in results:
        print(f"{r['label']:<20} {r['filter_scale']:>12} {r['precision']:>10.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['n_notes']:>7}")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()