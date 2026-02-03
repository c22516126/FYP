import numpy as np
import pretty_midi
from typing import Optional

def createNotes(
    pitchPost: np.ndarray,
    onsetPost: Optional[np.ndarray] = None,
    offsetPost: Optional[np.ndarray] = None,  # kept for future use
    pitchThreshold: float = 0.2,
    onsetThreshold: float = 0.5,
    minFrames: int = 3,
    bridgeGap: int = 2,
    sampleRate: int = 22050,
    fftHop: int = 512,
    midiOffset: int = 21,  # BasicPitch-style: 0 -> 21 (A0)
    velocity: int = 80,
):

    if pitchPost.ndim != 2:
        raise ValueError(f"pitchPost must be 2D (T, P), got shape {pitchPost.shape}")

    TIME, PITCH_BIN = pitchPost.shape
    secondsPerFrame = fftHop / sampleRate

    notes: list[pretty_midi.Note] = []

    for p in range(PITCH_BIN):
        inNote = False
        startFrame = None
        lastActive = None
        gap = 0

        for pitchTime in range(TIME):
            pitchValue = pitchPost[pitchTime, p]
            pitchActive = pitchValue >= pitchThreshold

            onsetActive = False
            if onsetPost is not None:
                onsetActive = onsetPost[pitchTime, p] >= onsetThreshold

            # ------------- STATE MACHINE -------------

            if not inNote:
                # Prefer onset-triggered starts
                if onsetPost is not None:
                    if onsetActive and pitchActive:
                        inNote = True
                        startFrame = pitchTime
                        lastActive = pitchTime
                        gap = 0
                else:
                    # Fallback: no onset info -> use pitch only
                    if pitchActive:
                        inNote = True
                        startFrame = pitchTime
                        lastActive = pitchTime
                        gap = 0
            else:
                # We are inside a note
                if pitchActive:
                    lastActive = pitchTime
                    gap = 0
                else:
                    gap += 1
                    if gap > bridgeGap:
                        # Close the note at lastActive + 1
                        endFrameExcl = (lastActive + 1) if lastActive is not None else pitchTime
                        length = endFrameExcl - startFrame
                        if length >= minFrames:
                            startTime = startFrame * secondsPerFrame
                            endTime = endFrameExcl * secondsPerFrame
                            midiPitch = p + midiOffset
                            notes.append(
                                pretty_midi.Note(
                                    velocity=velocity,
                                    pitch=midiPitch,
                                    start=startTime,
                                    end=endTime,
                                )
                            )
                        # reset state
                        inNote = False
                        startFrame = None
                        lastActive = None
                        gap = 0

        # If we end the sequence while still in a note
        if inNote and startFrame is not None and lastActive is not None:
            endFrameExcl = lastActive + 1
            length = endFrameExcl - startFrame
            if length >= minFrames:
                startTime = startFrame * secondsPerFrame
                endTime = endFrameExcl * secondsPerFrame
                midiPitch = p + midiOffset
                notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=midiPitch,
                        start=startTime,
                        end=endTime,
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

    test_notes = createNotes(mat, onsetPost=on, minFrames=2)
    print("Num notes:", len(test_notes))
    for n in test_notes:
        print("pitch", n.pitch, "start", n.start, "end", n.end)
