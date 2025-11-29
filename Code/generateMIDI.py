# generateMIDI.py

import pretty_midi

# ---------------------------------------------------------
# Polyphony Limiter
# ---------------------------------------------------------

def limit_polyphony(notes, max_poly=10):
    """
    Limit polyphony so that no more than max_poly notes overlap at once.
    Notes must be pretty_midi.Note objects.
    """

    # Sort notes by start time (primary) then end time
    notes_sorted = sorted(notes, key=lambda n: (n.start, n.end))

    filtered = []
    active = []

    for n in notes_sorted:
        # Remove notes that ended before this note starts
        active = [a for a in active if a.end > n.start]

        # If we have room, accept this new note
        if len(active) < max_poly:
            filtered.append(n)
            active.append(n)
        # Otherwise skip (too much polyphony)

    return filtered


# ---------------------------------------------------------
# MIDI Builder
# ---------------------------------------------------------

def buildMIDI(notes, output_path, instrument_name="Acoustic Grand Piano"):
    """
    Build a MIDI file from a list of pretty_midi.Note objects.
    Polyphony is automatically limited to avoid Fluidsynth crashes.
    """

    # FIRST: limit polyphony to avoid fluidsynth rvoice errors
    notes = limit_polyphony(notes, max_poly=10)

    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    for n in notes:
        instrument.notes.append(n)

    midi.instruments.append(instrument)
    midi.write(output_path)

    print(f"Saved MIDI with {len(notes)} notes → {output_path}")
