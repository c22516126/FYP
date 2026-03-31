import pretty_midi

def buildMIDI(noteEvents, outputPath, instrumentName="Acoustic Grand Piano"):
    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrumentName)
    )

    for start, end, pitch, amplitude in noteEvents:
        velocity = int(max(0, min(127, amplitude * 127)))  # convert amplitude to midi velocity

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(pitch),
            start=float(start),
            end=float(end),
        )

        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(outputPath)

    print(f"Saved MIDI with {len(noteEvents)} notes → {outputPath}")