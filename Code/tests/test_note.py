import numpy as np
from pipeline.noteCreation import createNotes

# Helper: convert notes -> (start_frame, end_frame, pitch)
def notes_to_frames(notes, hop=512, sr=22050):
    spf = hop / sr
    return [
        (int(note.start / spf), int(note.end / spf), note.pitch)
        for note in notes
    ]

# ==========================================================
# TEST 1 — Threshold crossing should produce one note
# ==========================================================
def test_single_note_simple():
    T, P = 50, 88
    pitchPost = np.zeros((T, P))

    pitchPost[10:21, 60] = 0.9

    notes = createNotes(
        pitchPost,
        pitch_threshold=0.2,
        min_frames=1
    )
    frames = notes_to_frames(notes)

    assert len(notes) == 1
    assert frames[0][0] == 10
    assert frames[0][1] == 21
    assert frames[0][2] == 60 + 21  # midi_offset

# ==========================================================
# TEST 2 — Gap bridging
# ==========================================================
def test_gap_bridging():
    T, P = 60, 88
    pitchPost = np.zeros((T, P))

    pitchPost[10:16, 40] = 0.9
    pitchPost[18:26, 40] = 0.9

    notes = createNotes(
        pitchPost,
        pitch_threshold=0.2,
        min_frames=1,
        bridge_gap=2
    )
    frames = notes_to_frames(notes)

    assert len(notes) == 1
    assert frames[0][0] == 10
    assert frames[0][1] == 26
    assert frames[0][2] == 40 + 21

# ==========================================================
# TEST 3 — Minimum note length
# ==========================================================
def test_minimum_length():
    T, P = 40, 88
    pitchPost = np.zeros((T, P))

    pitchPost[5:7, 30] = 0.9  # only 2 frames

    notes = createNotes(
        pitchPost,
        pitch_threshold=0.2,
        min_frames=3
    )

    assert len(notes) == 0

# ==========================================================
# TEST 4 — Separate notes must not merge
# ==========================================================
def test_two_separate_notes():
    T, P = 100, 88
    pitchPost = np.zeros((T, P))

    pitchPost[10:20, 50] = 0.9
    pitchPost[40:55, 50] = 0.9

    notes = createNotes(
        pitchPost,
        pitch_threshold=0.2,
        min_frames=1,
        bridge_gap=2
    )
    frames = notes_to_frames(notes)

    assert len(notes) == 2

    assert frames[0][0] == 10
    assert frames[0][1] == 20

    assert frames[1][0] == 40
    assert frames[1][1] == 55

# ==========================================================
# TEST 5 — Noise rejection
# ==========================================================
def test_noise_rejection():
    T, P = 100, 88
    noise = np.random.uniform(0.0, 0.15, (T, P))

    notes = createNotes(
        noise,
        pitch_threshold=0.2
    )

    assert len(notes) == 0

# ==========================================================
# TEST 6 — Sanity check output type
# ==========================================================
def test_sanity_shape():
    T, P = 500, 88
    pitchPost = np.random.uniform(0, 1, (T, P))

    notes = createNotes(pitchPost)
    assert isinstance(notes, list)
