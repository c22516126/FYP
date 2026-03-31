import pretty_midi
import numpy as np
import scipy.io.wavfile as wav

def midi_to_audio(midi_path, wav_path, soundfont="FluidR3_GM.sf2", sr=22050):
    """
    Simple MIDI → audio using pretty_midi + specified soundfont
    """

    pm = pretty_midi.PrettyMIDI(midi_path)

    # Render using your soundfont
    audio = pm.fluidsynth(fs=sr, sf2_path=soundfont)

    # Normalize safely
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    # Convert to float32
    audio = audio.astype(np.float32)

    # Save WAV
    wav.write(wav_path, sr, audio)

    print(f"[OK] Wrote: {wav_path}")