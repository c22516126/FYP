import pretty_midi
import numpy as np
import fluidsynth
from scipy.io.wavfile import write

def midi_to_audio(midi_path, wav_path, soundfont="FluidR3_GM.sf2", sr=44100, block_size=1024):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Synth setup
    fs = fluidsynth.Synth(samplerate=sr)
    sfid = fs.sfload(soundfont)
    if sfid == -1:
        raise ValueError(f"Failed to load soundfont: {soundfont}")
    fs.program_select(0, sfid, 0, 0)

    # Collect note events
    events = []
    for inst in pm.instruments:
        for n in inst.notes:
            events.append(("on", n.start, n.pitch, n.velocity))
            events.append(("off", n.end, n.pitch, n.velocity))

    events.sort(key=lambda x: x[1])
    next_event_idx = 0

    duration = pm.get_end_time()
    total_samples = int((duration + 1.0) * sr)
    audio = np.zeros(total_samples, dtype=np.float32)

    current_time = 0.0
    samples_written = 0

    # Convert stereo → mono
    def to_mono(block):
        block = np.asarray(block, dtype=np.float32)
        if block.size % 2 != 0:
            block = block[:-1]
        block = block.reshape(-1, 2).mean(axis=1)
        return block

    # Safe gain (lower = cleaner)
    GAIN = 0.2

    while samples_written < total_samples:
        block_end_time = current_time + block_size / sr

        # Process events in the block
        while (next_event_idx < len(events) and 
               events[next_event_idx][1] <= block_end_time):

            evt, t, pitch, vel = events[next_event_idx]
            sample_offset = int((t - current_time) * sr)

            # Render up to event
            if sample_offset > 0:
                block = fs.get_samples(sample_offset)
                block = to_mono(block) * GAIN
                block_len = len(block)
                audio[samples_written:samples_written + block_len] += block
                samples_written += block_len

            # Trigger MIDI event
            if evt == "on":
                fs.noteon(0, pitch, vel)
            else:
                fs.noteoff(0, pitch)

            current_time = t
            next_event_idx += 1

        # Render remaining block
        remaining = min(block_size, total_samples - samples_written)
        block = fs.get_samples(remaining)
        block = to_mono(block) * GAIN
        block_len = len(block)
        audio[samples_written:samples_written + block_len] += block

        samples_written += block_len
        current_time += block_len / sr

    # 🔥 Final normalization (prevents clipping)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9  # headroom

    write(wav_path, sr, audio)
    print(f"Saved audio → {wav_path}")


# Run directly
if __name__ == "__main__":
    midi_to_audio(
        midi_path="demo_output.mid",
        wav_path="output.wav",
        soundfont="FluidR3_GM.sf2"
    )
