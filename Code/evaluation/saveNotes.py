import pretty_midi

midiData = pretty_midi.PrettyMIDI(r'C:\Users\jason\school\FYP\FYP\Code\output\output.mid')
pianoData = midiData.instruments[0]
notes = pianoData.notes

filePath = "notes.txt"

with open(filePath, "w") as file:
    for note in notes:
        file.write(f"{note}\n")
    print(f"Saved into {filePath}")