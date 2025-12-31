import numpy as np
import pretty_midi
import fluidsynth

def midi_to_audio(midi_path, wav_path, soundfont="FluidR3_GM.sf2", sr=22050, block_size=512):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Initialize synth
    fs = fluidsynth.Synth(samplerate=sr)
    sfid = fs.sfload(soundfont)
    fs.program_select(0, sfid, 0, 0)

    # Collect all note on/off events
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            start_sample = int(note.start * sr)
            end_sample = int(note.end * sr)
            events.append((start_sample, "on", note.pitch, note.velocity))
            events.append((end_sample, "off", note.pitch, note.velocity))

    # Sort events by sample position
    events.sort(key=lambda x: x[0])

    total_samples = int(pm.get_end_time() * sr) + block_size
    audio = np.zeros(total_samples, dtype=np.float32)

    current_sample = 0
    event_index = 0

    while current_sample < total_samples:
        next_block_end = min(current_sample + block_size, total_samples)

        # Process all events that occur BEFORE the end of this block
        while event_index < len(events) and events[event_index][0] < next_block_end:
            event_sample, etype, pitch, vel = events[event_index]

            # Render up to the event
            delta = event_sample - current_sample
            if delta > 0:
                block = fs.get_samples(delta)
                block = np.asarray(block, dtype=np.float32)
                block = block.reshape(-1, 2).mean(axis=1)
                audio[current_sample : current_sample + delta] = block
                current_sample += delta

            # Apply the event
            if etype == "on":
                fs.noteon(0, pitch, vel)
            else:
                fs.noteoff(0, pitch)

            event_index += 1

        # Render the rest of the block
        remaining = next_block_end - current_sample
        if remaining > 0:
            block = fs.get_samples(remaining)
            block = np.asarray(block, dtype=np.float32)
            block = block.reshape(-1, 2).mean(axis=1)
            audio[current_sample : current_sample + remaining] = block
            current_sample += remaining

    # Normalize audio (safely)
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp

    # Save WAV
    import scipy.io.wavfile as wav
    wav.write(wav_path, sr, audio)

    print(f"[OK] Wrote: {wav_path}")
