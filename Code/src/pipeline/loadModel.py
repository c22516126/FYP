# load model for use during inference

import tensorflow as tf
import os

def loadModel(modelPath):
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"Model not found at: {modelPath}")

    interpreter = tf.lite.Interpreter(model_path=str(modelPath)) # create an instance of tf.lite.Interpreter, load model into it
    interpreter.allocate_tensors() # find tensors needed, determine shape, allocate memory, set execution order
    return interpreter