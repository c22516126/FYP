import numpy as np
from noteCreation import createNotes

# Utility: convert notes -> (start_frame, end_frame, pitch)
def notes_to_frames(notes, hop=512, sr=22050):
    spf = hop / sr
    return [
        (int(note.start / spf), int(note.end / spf), note.pitch)
        for note in notes
    ]

# ==========================================================
# TEST 1 — Basic threshold crossing should produce one note
# ==========================================================
def test_single_note_simple():
    T = 50
    P = 88
    pitchPost = np.zeros((T, P))

    # Activate pitch 60 from frame 10 → 20
    pitchPost[10:21, 60] = 0.9

    notes = createNotes(pitchPost, threshold=0.2, min_frames=1)
    frames = notes_to_frames(notes)

    assert len(notes) == 1, "Should detect exactly one note"
    assert frames[0][0] == 10, "Start frame incorrect"
    assert frames[0][1] == 21, "End frame incorrect"
    assert frames[0][2] == 60, "Pitch incorrect"


# ==========================================================
# TEST 2 — Fragmentation unless we bridge small gaps
# ==========================================================
def test_gap_bridging():
    T = 60
    P = 88
    pitchPost = np.zeros((T, P))

    # Active 10–15 then gap 16–17 then active 18–25
    pitchPost[10:16, 40] = 0.9
    pitchPost[18:26, 40] = 0.9

    # bridge_gap=2 should merge this into ONE note
    notes = createNotes(
        pitchPost, threshold=0.2,
        bridge_gap=2, min_frames=1
    )
    frames = notes_to_frames(notes)

    assert len(notes) == 1, "Two segments within gap should merge into one"
    assert frames[0][0] == 10
    assert frames[0][1] == 26


# ==========================================================
# TEST 3 — Ensure small notes are removed by min_frames
# ==========================================================
def test_minimum_length():
    T = 40
    P = 88
    pitchPost = np.zeros((T, P))

    # Only 2 frames active (below min_frames=3)
    pitchPost[5:7, 30] = 0.9

    notes = createNotes(pitchPost, threshold=0.2, min_frames=3)
    assert len(notes) == 0, "Short notes must be filtered out"


# ==========================================================
# TEST 4 — Two distinct notes should not merge
# ==========================================================
def test_two_separate_notes():
    T = 100
    P = 88
    pitchPost = np.zeros((T, P))

    # First note
    pitchPost[10:20, 50] = 0.9
    # Big gap
    pitchPost[40:55, 50] = 0.9

    notes = createNotes(pitchPost, threshold=0.2, bridge_gap=2)
    frames = notes_to_frames(notes)

    assert len(notes) == 2, "Separate notes must NOT be merged"
    assert frames[0][0] == 10 and frames[0][1] == 20
    assert frames[1][0] == 40 and frames[1][1] == 55


# ==========================================================
# TEST 5 — Random noise below threshold must not create notes
# ==========================================================
def test_noise_rejection():
    T = 100
    P = 88

    # Noise from 0.0 to 0.15 (below threshold)
    noise = np.random.uniform(0.0, 0.15, size=(T, P))

    notes = createNotes(noise, threshold=0.2)
    assert len(notes) == 0, "Noise below threshold must produce no notes"


# ==========================================================
# TEST 6 — Test real posteriorgram-like shape (sanity test)
# ==========================================================
def test_sanity_shape():
    T = 500
    P = 88
    pitchPost = np.random.uniform(0, 1, (T, P))

    notes = createNotes(pitchPost)

    # Not testing correctness here — just sanity
    assert isinstance(notes, list), "Output should be a list"
