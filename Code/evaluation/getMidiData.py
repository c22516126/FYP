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

def makeAnnotationFile(inputPath, outputPath):
    notes = getNoteData(inputPath)

