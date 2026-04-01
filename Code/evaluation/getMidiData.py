import numpy as np
import pretty_midi
import mir_eval

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

def noteToEvalData(notes):
    intervals = []
    pitches = []

    for start, end, pitch, _ in notes:
        intervals.append([start, end])
        pitches.append(mir_eval.util.midi_to_hz(pitch))

    return np.array(intervals), np.array(pitches)