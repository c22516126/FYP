# model_loader.py

import tensorflow as tf
import os

def loadTfliteModel(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter