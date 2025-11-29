import pretty_midi
import numpy as np
from scipy.io.wavfile import write
import fluidsynth
import time

def midi_to_audio(midi_path, wav_path, soundfont="FluidR3_GM.sf2", sr=44100):
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    fs = fluidsynth.Synth(samplerate=sr)
    sfid = fs.sfload(soundfont)
    fs.program_select(0, sfid, 0, 0)
    fs.start()

    # Determine end time of the MIDI
    duration = pm.get_end_time()
    total_samples = int(duration * sr)

    audio = np.zeros(total_samples, dtype=np.float32)

    # Play ALL notes in the timeline
    for inst in pm.instruments:
        for note in inst.notes:
            start = int(note.start * sr)
            end = int(note.end * sr)

            fs.noteon(0, note.pitch, note.velocity)
            samples = fs.get_samples(end - start)
            fs.noteoff(0, note.pitch)

            audio[start:end] += samples[:end - start]

    write(wav_path, sr, audio)


if __name__ == "__main__":
    midi_to_audio("demo_output.mid", "demo_output.wav")
