# inference.py

import numpy as np

def runModel(interpreter, audioWindow):
    """
    Runs the TFLite BasicPitch model on a single audio window.

    Args:
        interpreter: tf.lite.Interpreter (already allocated)
        audioWindow: numpy array of shape (1, frames, 1)

    Returns:
        pitchPost  (frames, 88)
        onsetPost  (frames, 88)
        offsetPost (frames, 88)
    """
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()

    # Feed audio window into model
    interpreter.set_tensor(inputDetails[0]["index"], audioWindow.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Retrieve outputs
    pitch = interpreter.get_tensor(outputDetails[0]["index"])
    onset = interpreter.get_tensor(outputDetails[1]["index"])
    offset = interpreter.get_tensor(outputDetails[2]["index"])

    # Remove batch dim: (1, frames, 88) → (frames, 88)
    pitch = np.squeeze(pitch, axis=0)
    onset = np.squeeze(onset, axis=0)
    offset = np.squeeze(offset, axis=0)

    return pitch, onset, offset



# -----------------------------------------------------------------------
# FULL INFERENCE SPEED TEST (SAFE - DOES NOT AFFECT MAIN)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from loadModel import loadTfliteModel
    from basic_pitch.inference import get_audio_input

    AUDIO_FILE = "testInput/Avril14th.mp3"
    MODEL_PATH = (
        "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/code/venv/"
        "Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"
    )

    print("Running inference speed test...")
    interpreter = loadTfliteModel(MODEL_PATH)

    windowCount = 0
    start = time.time()

    # Test EXACT setup your main uses.
    for audioWindow, _, _ in get_audio_input(AUDIO_FILE, overlap_len=30, hop_size=512):
        pitch, onset, offset = runModel(interpreter, audioWindow)
        windowCount += 1

    end = time.time()

    print(f"\n✔ Inference finished for {windowCount} windows")
    print("Total time:", end - start, "seconds")
    print("Avg per window:", (end - start) / windowCount)
