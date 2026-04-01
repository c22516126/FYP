import pretty_midi

def saveNotes(inputPath, outputPath):
    midiData = pretty_midi.PrettyMIDI(inputPath)
    pianoData = midiData.instruments[0]
    notes = pianoData.notes

    filePath = outputPath

    notes = pianoData.notes

    seen = set()
    duplicates = 0

    for n in notes:
        key = (round(n.start,5), round(n.end,5), n.pitch)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    print("Total notes:", len(notes))
    print("Duplicate notes:", duplicates)
    print("Unique notes:", len(seen))

    pedal_events = []
    for cc in pianoData.control_changes:
        if cc.number == 64:   # sustain pedal
            pedal_events.append((cc.time, cc.value))

    with open(filePath, "w") as file:
        for note in notes:
            file.write(f"{note}\n")
        file.write(f"End time:{midiData.get_end_time()}\n")
        file.write(f"tempo changes:{midiData.get_tempo_changes()}")
        print(f"Saved into {filePath}")

     # write pedal info
        if pedal_events:
            file.write("\nSustain pedal events:\n")
            for time, value in pedal_events:
                file.write(f"time={time:.3f}, value={value}\n")
        else:
            file.write("\nNo sustain pedal events found\n")

path = r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\CSM\CSM.mid'
outpath = r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\CSM\CSMnotes.txt'
saveNotes(path, outpath)