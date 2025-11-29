# model_loader.py

import tensorflow as tf
import os

def loadTfliteModel(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# -----------------------------------------------------------------------
# MODEL LOADING SPEED TEST (SAFE - DOES NOT AFFECT MAIN)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import time

    MODEL_PATH = "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/code/venv/Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"

    print("Testing model loader...")
    start = time.time()

    interpreter = loadTfliteModel(MODEL_PATH)

    end = time.time()

    print("✔ Model loaded successfully")
    print("Load time:", end - start, "seconds")
