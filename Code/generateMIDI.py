import pretty_midi

def buildMIDI(notes, outfile="output.mid"):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for pitch, start, end in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=start,
                end=end
            )
        )

    pm.instruments.append(instrument)
    pm.write(outfile)
    return pm

if __name__ == "__main__":
    # Fake notes: middle C from 0.0 → 1.0
    notes = [(60, 0.0, 1.0)]

    pm = buildMIDI(notes, "test_midi.mid")
    print("Generated MIDI with", len(pm.instruments[0].notes), "notes")

    assert len(pm.instruments) == 1
    assert len(pm.instruments[0].notes) == 1
    assert pm.instruments[0].notes[0].pitch == 60

    print("✔ midi_builder.py passed basic sanity test")