import numpy as np
import tensorflow as tf
from basic_pitch.inference import get_audio_input
import pretty_midi

AUDIO_FILE = "testInput/Avril14th.mp3"
MODEL_PATH = "C:/Users/jason/Desktop/School/y4s1/FYP/final year project/code/venv/Lib/site-packages/basic_pitch/saved_models/icassp_2022/nmp.tflite"


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_model(interpreter, audio_window):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], audio_window)
    interpreter.invoke()

    pitch = interpreter.get_tensor(output_details[0]["index"])
    onset = interpreter.get_tensor(output_details[1]["index"])
    offset = interpreter.get_tensor(output_details[2]["index"])
    return pitch, onset, offset


def posteriorgram_to_midi(pitch_post, hop=512, sr=22050, outfile="demo_output.mid"):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    frames, pitches = pitch_post.shape
    seconds_per_frame = hop / sr

    for p in range(pitches):
        active = False
        start_time = 0.0

        for t in range(frames):
            prob = pitch_post[t, p]

            if prob > 0.2 and not active:
                active = True
                start_time = t * seconds_per_frame

            elif prob <= 0.2 and active:
                end_time = t * seconds_per_frame
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=p,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
                active = False

        if active:
            end_time = frames * seconds_per_frame
            note = pretty_midi.Note(
                velocity=80,
                pitch=p,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(outfile)
    print("🎵 Wrote MIDI file:", outfile)


def main():
    print("🔊 Loading audio:", AUDIO_FILE)
    gen = get_audio_input(AUDIO_FILE, overlap_len=30, hop_size=512)
    audio_windowed, _, _ = next(gen)
    print("Audio window shape:", audio_windowed.shape)  # (1, 43844, 1)

    # Load the TFLite model
    interpreter = load_tflite_model(MODEL_PATH)

    # Check model input shape
    input_details = interpreter.get_input_details()
    print("Model expects shape:", input_details[0]['shape'])

    # Input is already 3D, use as-is
    model_input = audio_windowed.astype(np.float32)

    # Run model
    pitch_post, onset_post, offset_post = run_model(interpreter, model_input)

    # Remove batch dimension
    pitch_post = np.squeeze(pitch_post, axis=0)
    onset_post = np.squeeze(onset_post, axis=0)
    offset_post = np.squeeze(offset_post, axis=0)

    print("Posteriorgram shapes:")
    print("  Pitch: ", pitch_post.shape)
    print("  Onset: ", onset_post.shape)
    print("  Offset:", offset_post.shape)

    posteriorgram_to_midi(pitch_post)
    print("✅ Demo complete.")


if __name__ == "__main__":
    main()
