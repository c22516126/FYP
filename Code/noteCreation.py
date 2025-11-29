def createNotes(pitch_post, threshold=0.2, hop=512, sr=22050):
    notes = []
    frames, pitches = pitch_post.shape
    spf = hop / sr  # seconds per frame

    for p in range(pitches):
        active = False
        start = 0.0

        for t in range(frames):
            prob = pitch_post[t, p]

            if prob > threshold and not active:
                active = True
                start = t * spf

            elif prob <= threshold and active:
                end = t * spf
                notes.append((p, start, end))
                active = False

        if active:
            end = frames * spf
            notes.append((p, start, end))

    return notes

if __name__ == "__main__":
    import numpy as np

    # Fake posteriorgram: pitch 60 active from frames 10–20
    fake = np.zeros((50, 88))
    fake[10:20, 60] = 0.9

    notes = createNotes(fake)
    print("Test Notes:", notes)

    assert len(notes) == 1, "Should detect exactly 1 note"
    pitch, start, end = notes[0]
    assert pitch == 60, "Pitch should be 60"
    assert start > 0, "Start time should be > 0"
    assert end > start, "End should be after start"

    print("✔ note_extraction.py passed basic sanity test")