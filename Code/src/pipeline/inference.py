import numpy as np

# -------------------------
# Signature-runner inference
# -------------------------

def runModelFast(interpreter, audioWindow):
    runner = interpreter.get_signature_runner()

    outputs = runner(input_2=audioWindow.astype(np.float32))

    # Older TFLite models use these keys:
    pitch = outputs["note"]      # pitch posteriorgram
    onset = outputs["onset"]     # onset posteriorgram
    offset = outputs["contour"]  # pitch contour (offset)

    # squeeze batch dim
    pitch = np.squeeze(pitch, axis=0)
    onset = np.squeeze(onset, axis=0)
    offset = np.squeeze(offset, axis=0)

    return pitch, onset, offset

# remove overlap and build posteriorgram
def unwrapOutput(batchedOutput, framesPerWindow, framesPerStride):
    """
    Stitch overlapping model outputs while preserving
    correct time alignment, including start/end boundaries.
    """

    if batchedOutput.ndim != 3:
        return batchedOutput

    stitched = []

    center_start = (framesPerWindow - framesPerStride) // 2
    center_end = center_start + framesPerStride

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

    return np.vstack(stitched)


# debugging

def dump_pitch_trace(
    pitchPost,
    onsetPost,
    pitch_idx,
    sr,
    fft_hop,
    start_sec=0.0,
    end_sec=10.0,
    path="pitch_trace.csv",
):
    start_frame = int(start_sec * sr / fft_hop)
    end_frame = int(end_sec * sr / fft_hop)

    with open(path, "w") as f:
        f.write("frame,time_sec,pitch_prob,onset_prob\n")
        for t in range(start_frame, min(end_frame, pitchPost.shape[0])):
            time_sec = t * fft_hop / sr
            pitch_prob = pitchPost[t, pitch_idx]
            onset_prob = onsetPost[t, pitch_idx] if onsetPost is not None else 0.0
            f.write(
                f"{t},{time_sec:.6f},"
                f"{pitch_prob:.4f},{onset_prob:.4f}\n"
            )

def dump_frame_slice(pitchPost, onsetPost, frame, path="frame_slice.csv"):
    with open(path, "w") as f:
        f.write("pitch,pitch_prob,onset_prob\n")
        for p in range(pitchPost.shape[1]):
            f.write(
                f"{p},"
                f"{pitchPost[frame, p]:.4f},"
                f"{onsetPost[frame, p]:.4f}\n"
            )
