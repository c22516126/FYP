import pretty_midi
import numpy as np

def buildMIDI(noteEvents, outputPath, instrumentName="Acoustic Grand Piano"):
 
    midi = pretty_midi.PrettyMIDI() # initialize MIDI object

    # create instrument
    instrument = pretty_midi.Instrument( 
        program=pretty_midi.instrument_name_to_program(instrumentName)
    )

    for start, end, pitch, amplitude in noteEvents:
        velocity = int(np.round(127 * amplitude)) # map amplitude value to MIDI velocity range

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(pitch),
            start=float(start),
            end=float(end),
        )

        instrument.notes.append(note) # add note to instrument

    midi.instruments.append(instrument) # add instrument to MIDI
    midi.write(outputPath)

    print(f"Saved MIDI with {len(noteEvents)} notes -> {outputPath}")
