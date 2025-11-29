# inference.py

import numpy as np


def runModel(interpreter, audioWindow):
    """
    Runs the BasicPitch TFLite model on a single audio window.

    Args:
        interpreter: tf.lite.Interpreter (already allocated)
        audioWindow: numpy array of shape (1, frames, 1)

    Returns:
        pitchPost  (frames, 88)
        onsetPost  (frames, 88)
        offsetPost (frames, 88)
    """

    # Try to use TFLite signature runner (faster, like BasicPitch)
    runner = getattr(interpreter, "_signature_runner", None)

    if runner is None:
        # First time: create and cache it on the interpreter
        try:
            runner = interpreter.get_signature_runner()
            interpreter._signature_runner = runner
        except AttributeError:
            # Fallback: old-style manual invoke (slower)
            inputDetails = interpreter.get_input_details()
            outputDetails = interpreter.get_output_details()

            interpreter.set_tensor(inputDetails[0]["index"], audioWindow.astype(np.float32))
            interpreter.invoke()

            pitch = interpreter.get_tensor(outputDetails[0]["index"])
            onset = interpreter.get_tensor(outputDetails[1]["index"])
            offset = interpreter.get_tensor(outputDetails[2]["index"])

            pitch = np.squeeze(pitch, axis=0)
            onset = np.squeeze(onset, axis=0)
            offset = np.squeeze(offset, axis=0)

            return pitch, onset, offset

    # Signature runner path (BasicPitch-style)
    # BasicPitch TFLite signatures:
    #   input key: "input_2"
    #   output keys: "note", "onset", "contour"
    outputs = runner(input_2=audioWindow.astype(np.float32))

    pitch = outputs["note"][0]     # (1, T, 88) → (T, 88)
    onset = outputs["onset"][0]
    offset = outputs["contour"][0]

    return pitch, onset, offset


# -----------------------------------------------------------------------
# FULL INFERENCE SPEED TEST (SAFE - DOES NOT AFFECT MAIN)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from loadModel import loadTfliteModel
    from basic_pitch.inference import get_audio_input
    from basic_pitch.constants import AUDIO_N_SAMPLES, FFT_HOP

    AUDIO_FILE = "testInput/Avril14th.mp3"
    MODEL_PATH = (
        "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/code/venv/"
        "Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"
    )

    # This matches BasicPitch's overlap logic:
    #   n_overlapping_frames = 30
    #   overlap_len = n_overlapping_frames * FFT_HOP
    #   hop_size = AUDIO_N_SAMPLES - overlap_len
    nOverlappingFrames = 30
    overlapLen = nOverlappingFrames * FFT_HOP
    hopSize = AUDIO_N_SAMPLES - overlapLen

    print("\n=== Batched-style Inference Speed Test (BasicPitch settings) ===")

    # 1. Load model
    t0 = time.time()
    interpreter = loadTfliteModel(MODEL_PATH)
    t1 = time.time()
    print(f"[1] Model load time: {t1 - t0:.4f} sec")

    # 2. Collect windows using the SAME hop/overlap as BasicPitch
    audioWindows = []
    t2 = time.time()
    for win, _, _ in get_audio_input(AUDIO_FILE, overlapLen, hopSize):
        audioWindows.append(win)
    t3 = time.time()

    print(f"[2] Windows collected: {len(audioWindows)}")
    print(f"[2] Window prep time: {t3 - t2:.4f} sec")

    # 3. Run inference window-by-window, but:
    #    - far fewer windows thanks to large hop size
    #    - faster per-call using signature runner
    t4 = time.time()
    totalFrames = 0
    for win in audioWindows:
        pitchPost, onsetPost, offsetPost = runModel(interpreter, win)
        totalFrames += pitchPost.shape[0]
    t5 = time.time()

    print(f"[3] Inference time over all windows: {t5 - t4:.4f} sec")
    print(f"[3] Total frames processed: {totalFrames}")

    print(f"[TOTAL] Full test time: {t5 - t0:.4f} sec\n")
