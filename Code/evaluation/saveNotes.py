import pretty_midi

def saveNotes(inputPath, outputPath):
    midiData = pretty_midi.PrettyMIDI(inputPath)
    pianoData = midiData.instruments[0]
    notes = pianoData.notes

    filePath = outputPath

    with open(filePath, "w") as file:
        for note in notes:
            file.write(f"{note}\n")
        file.write(f"End time:{midiData.get_end_time()}\n")
        file.write(f"tempo changes:{midiData.get_tempo_changes()}")
        print(f"Saved into {filePath}")