import numpy as np

# build posteriorgram for the whole audio
def unwrapOutput(batchedOutput, framesPerWindow, framesPerStride):

    # only run on expected shape (windows, frames, pitches)
    assert batchedOutput.ndim == 3, f"Expected (windows, frames, pitches), got shape {batchedOutput.shape}"

    stitched = []

    # keep center, remove edges
    centreStart = (framesPerWindow - framesPerStride) // 2
    centreEnd = centreStart + framesPerStride

    # detect first/last window
    nWindows = batchedOutput.shape[0]

    for i, window in enumerate(batchedOutput):

        # first window - keep from start
        if i == 0: 
            stitched.append(window[:centreEnd])

        # last window - keep until end
        elif i == nWindows - 1:
            stitched.append(window[centreStart:])

        # middle windows - keep center slice only
        else:
            stitched.append(window[centreStart:centreEnd])

    # stack center slices to get final posteriorgram
    return np.vstack(stitched)