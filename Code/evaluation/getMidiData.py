import numpy as np
import pretty_midi
import mir_eval
import os

def getNoteData(path):
    midiData = pretty_midi.PrettyMIDI(path)
    pianoData = midiData.instruments[0]
    notes = pianoData.notes

    return notes

# get 2d array of note intervals
def getIntervals(path):
    notes = getNoteData(path)
    intervals = np.array([[note.start, note.end] for note in notes])

    return intervals

def getShiftedIntervals(path, shift):
    notes = getNoteData(path)
    intervals = np.array([[note.start - shift, note.end - shift] for note in notes])

    return intervals

def shiftIntervals(intervals, shift):
    shifted = []

    for start, end in intervals:
        new_start = start + shift
        new_end = end + shift

        if new_end <= 0:
            continue

        shifted.append([max(0, new_start), max(0, new_end)])

    return np.array(shifted)
        
# get 1d array of pitches in HZ
def getPitchesInHZ(path):
    notes = getNoteData(path)
    pitches = []
    for note in notes:
        pitchHZ = mir_eval.util.midi_to_hz(note.pitch) # convert from MIDI to HZ
        pitches.append(pitchHZ)

    pitches = np.array(pitches) # convert to numpy
    return pitches

def getPitchesInMIDI(path):
    notes = getNoteData(path)
    pitches = []
    for note in notes:
        pitches.append(note.pitch)

    pitches = np.array(pitches) # convert to numpy
    return pitches

def buildDataset(eval_folder):
    dataset = []

    for file in os.listdir(eval_folder):
        if file.endswith(".mid"):
            base = file[:-4]  # remove .mid
            midi_path = os.path.join(eval_folder, file)
            audio_path = os.path.join(eval_folder, base + ".mp3")

            if os.path.exists(audio_path):
                dataset.append((audio_path, midi_path))
            else:
                print(f"Warning: Missing audio for {file}")

    return dataset

def noteToEvalData(notes):
    intervals = []
    pitches = []

    for start, end, pitch, _ in notes:
        intervals.append([start, end])
        pitches.append(mir_eval.util.midi_to_hz(pitch))

    return np.array(intervals), np.array(pitches)


# save MIDI information into text format
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