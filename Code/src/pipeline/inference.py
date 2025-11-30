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

# -------------------------
# Fast unwrapping like BP
# -------------------------

def unwrapOutput(batchedOutput, audioLen, nOverlapFrames, hopSize):
    """
    Removes overlap and rebuilds full posteriorgram.
    """

    if len(batchedOutput.shape) != 3:
        return batchedOutput

    nHalf = nOverlapFrames // 2
    if nHalf > 0:
        batchedOutput = batchedOutput[:, nHalf:-nHalf, :]

    # Flatten windowed output -> (totalFrames, 88)
    B, F, P = batchedOutput.shape
    stitched = batchedOutput.reshape(B * F, P)

    # How many frames expected?
    expected = int(audioLen / 512)

    return stitched[:expected]
