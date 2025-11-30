import numpy as np
import pretty_midi
from typing import Optional


def createNotes(
    pitchPost: np.ndarray,
    onsetPost: Optional[np.ndarray] = None,
    offsetPost: Optional[np.ndarray] = None,  # kept for future use
    pitch_threshold: float = 0.3,
    onset_threshold: float = 0.5,
    min_frames: int = 3,
    bridge_gap: int = 2,
    sr: int = 22050,
    fft_hop: int = 512,
    midi_offset: int = 21,  # BasicPitch-style: 0 -> 21 (A0)
    velocity: int = 80,
):
    """
    Onset-aware, threshold-based note extraction.

    Args:
        pitchPost:  (T, P) pitch posteriorgram
        onsetPost:  (T, P) onset posteriorgram (optional but recommended)
        offsetPost: unused for now, reserved for future refinement
        pitch_threshold: min prob. to consider pitch active
        onset_threshold: min prob. to treat as onset (if onsetPost is given)
        min_frames: minimum length in frames
        bridge_gap: max inactive frames allowed inside a note
        sr: sample rate of original audio
        fft_hop: hop size of spectrogram / model in samples (NOT window hop)
        midi_offset: MIDI note for pitch bin 0 (21 for A0)
        velocity: fixed MIDI velocity for all created notes
    """

    if pitchPost.ndim != 2:
        raise ValueError(f"pitchPost must be 2D (T, P), got shape {pitchPost.shape}")

    T, P = pitchPost.shape
    seconds_per_frame = fft_hop / sr

    notes: list[pretty_midi.Note] = []

    for p in range(P):
        in_note = False
        start_frame = None
        last_active = None
        gap = 0

        for t in range(T):
            pitch_val = pitchPost[t, p]
            pitch_active = pitch_val >= pitch_threshold

            onset_active = False
            if onsetPost is not None:
                onset_active = onsetPost[t, p] >= onset_threshold

            # ------------- STATE MACHINE -------------

            if not in_note:
                # Prefer onset-triggered starts
                if onsetPost is not None:
                    if onset_active and pitch_active:
                        in_note = True
                        start_frame = t
                        last_active = t
                        gap = 0
                else:
                    # Fallback: no onset info -> use pitch only
                    if pitch_active:
                        in_note = True
                        start_frame = t
                        last_active = t
                        gap = 0
            else:
                # We are inside a note
                if pitch_active:
                    last_active = t
                    gap = 0
                else:
                    gap += 1
                    if gap > bridge_gap:
                        # Close the note at last_active + 1
                        end_frame_excl = (last_active + 1) if last_active is not None else t
                        length = end_frame_excl - start_frame
                        if length >= min_frames:
                            start_time = start_frame * seconds_per_frame
                            end_time = end_frame_excl * seconds_per_frame
                            midi_pitch = p + midi_offset
                            notes.append(
                                pretty_midi.Note(
                                    velocity=velocity,
                                    pitch=midi_pitch,
                                    start=start_time,
                                    end=end_time,
                                )
                            )
                        # reset state
                        in_note = False
                        start_frame = None
                        last_active = None
                        gap = 0

        # If we end the sequence while still in a note
        if in_note and start_frame is not None and last_active is not None:
            end_frame_excl = last_active + 1
            length = end_frame_excl - start_frame
            if length >= min_frames:
                start_time = start_frame * seconds_per_frame
                end_time = end_frame_excl * seconds_per_frame
                midi_pitch = p + midi_offset
                notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=midi_pitch,
                        start=start_time,
                        end=end_time,
                    )
                )

    # Sort by time then pitch for sanity
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


# Quick self-test when run directly
if __name__ == "__main__":
    T, P = 100, 88
    mat = np.zeros((T, P), dtype=np.float32)
    on = np.zeros((T, P), dtype=np.float32)

    # Fake: one note around pitch bin 60 for frames 10–30
    mat[10:31, 60] = 0.9
    on[10, 60] = 0.9

    test_notes = createNotes(mat, onsetPost=on, min_frames=2)
    print("Num notes:", len(test_notes))
    for n in test_notes:
        print("pitch", n.pitch, "start", n.start, "end", n.end)
