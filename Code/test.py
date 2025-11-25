import numpy as np
import tensorflow as tf
from basic_pitch.inference import get_audio_input

# Path to a short audio file (preferably < 5 seconds)
AUDIO_FILE = "testInput/Avril14th.mp3"
ICASSP_2022_MODEL_PATH = "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/venv/Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"


def main():
    print("Step 1: Preprocessing audio...")
    
    # Get audio input (returns a generator), so we use next() to get the first windowed chunk
    audio_windowed, _, audio_original_length = next(get_audio_input(AUDIO_FILE, overlap_len=30, hop_size=512))
    
    print("Audio window shape:", audio_windowed.shape)  # Now we can access shape, since it's a numpy array

    # Convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(audio_windowed, dtype=tf.float32)
    
    print("Input tensor shape:", input_tensor.shape)

    print("Step 2: Loading model...")
    interpreter = tf.lite.Interpreter(model_path=ICASSP_2022_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Step 3: Running inference...")
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    pitch_probs = interpreter.get_tensor(output_details[0]["index"])
    onset_probs = interpreter.get_tensor(output_details[1]["index"])
    offset_probs = interpreter.get_tensor(output_details[2]["index"])

    print("Pitch probs shape:", pitch_probs.shape)
    print("Onset probs shape:", onset_probs.shape)
    print("Offset probs shape:", offset_probs.shape)

    # Print a small slice to verify it's not all zeros
    print("Pitch probs (frame 0):", pitch_probs[0][:10])
    print("Onset probs (frame 0):", onset_probs[0][:10])
    print("Offset probs (frame 0):", offset_probs[0][:10])

if __name__ == "__main__":
    main()
