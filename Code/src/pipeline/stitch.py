import numpy as np

# build posteriorgram for the whole audio
def unwrapOutput(batchedOutput, framesPerWindow, framesPerStride):

    # only run on expected shape (windows, frames, pitches)
    if batchedOutput.ndim != 3:
        return batchedOutput

    stitched = []

    # keep center, remove edges
    center_start = (framesPerWindow - framesPerStride) // 2
    center_end = center_start + framesPerStride

    # detect first/last window
    num_windows = batchedOutput.shape[0]

    for i, window in enumerate(batchedOutput):
        if i == 0:
            # FIRST window: keep from start
            stitched.append(window[:center_end])

        elif i == num_windows - 1:
            # LAST window: keep until end
            stitched.append(window[center_start:])

        else:
            # MIDDLE windows: keep center slice only
            stitched.append(window[center_start:center_end])

    # stack center slices to get final posteriorgram
    return np.vstack(stitched)
