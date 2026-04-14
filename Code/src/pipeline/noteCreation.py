import numpy as np
import scipy
from typing import List, Tuple

MAX_FREQUENCY_INDEX = 87
MIDI_OFFSET = 21

def createNotes(
    frames: np.array, # energy (is pitch active during this frame)
    onsets: np.array,
    onsetThreshold: float = 0.5,
    frameThreshold: float = 0.3,
    minimumNoteLength: int = 11,
    energyTolerance: int = 8,
    melodia: bool = True
) ->List[Tuple[int,int,int,float]]:

    nFrames = frames.shape[0]
    
    # peak pick in onset matrix
    peakThresholdMatrix = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis = 0) # find the index of values that are greater than its neighbours
    peakThresholdMatrix[peaks] = onsets[peaks]

    # Thresholding step - find all positions (index) where onset value is over threshold
    onsetIndex = np.where(peakThresholdMatrix >= onsetThreshold)

    # reverse onset pitch and time arrays - goes hand in hand for energy locking, lets later notes lock energy first
    onsetTimeIndex = onsetIndex[0][::-1]
    onsetFreqIndex = onsetIndex[1][::-1]

    # prevent duplicate notes by locking time/frequency energy once claimed by a note
    # create mutable copy of frame posteriorgram
    remainingEnergy = np.zeros(frames.shape)
    remainingEnergy[:, :] = frames[:, :]

    # for all valid onset candidates
    noteEvents = [] # make note array
    for noteStartIndex, frequencyIndex in zip(onsetTimeIndex, onsetFreqIndex): # pair reversed note and frequency array, loop through
        if noteStartIndex >= nFrames -1: # skip onsets at end of audio
            continue

        # find time index
        i = noteStartIndex + 1
        k = 0 # number of frames since energy has dropped below energy threshold

        # go forward until energy dies out (tolerance wise)
        # while still inside the audio and above the energy loss threshold
        while (i < nFrames - 1 and  k < energyTolerance):
            if remainingEnergy[i, frequencyIndex] < frameThreshold:
                k += 1
            else: # reset counter if remaining energy passes threshold
                k = 0
            i += 1

        i -= k # rewind k to last frame where energy is above the threshold
    
        if (i - noteStartIndex <= minimumNoteLength): # skip if note is too small
            continue

        # lock energy once claimed by a note
        remainingEnergy[noteStartIndex:i, frequencyIndex] = 0 # remove energy at main pitch
        if frequencyIndex < MAX_FREQUENCY_INDEX:
            remainingEnergy[noteStartIndex:i, frequencyIndex + 1] = 0 # remove energy in neighbour above
        if frequencyIndex > 0:
            remainingEnergy[noteStartIndex:i, frequencyIndex - 1] = 0 # remove energy in neighbour below

        # create the note
        amplitude = np.mean(frames[noteStartIndex:i, frequencyIndex]) # normalized loudness estimate
        noteEvents.append(
            (
                noteStartIndex, # note start
                i, # note end
                frequencyIndex + MIDI_OFFSET, # pitch
                amplitude # velocity
            )
        )

    # melodia trick (create notes out of strongest remaining energy peaks)
    if melodia:
        energyShape = remainingEnergy.shape

        while (np.max(remainingEnergy) > frameThreshold): # while energy is strong enough to be considered a note
            noteCentre, frequencyIndex = np.unravel_index(np.argmax(remainingEnergy), energyShape)
            remainingEnergy[noteCentre, frequencyIndex] = 0

            # forward pass
            i = noteCentre + 1
            k = 0

            # go forward until end of audio or energy is below tolerance
            while (i < nFrames - 1 and k < energyTolerance):
                if (remainingEnergy[i, frequencyIndex] < frameThreshold):
                    k += 1
                else:
                    k = 0

                # lock energy for pitch bin and neighbouring pitch bins
                remainingEnergy[i, frequencyIndex] = 0
                if (frequencyIndex < MAX_FREQUENCY_INDEX):
                    remainingEnergy[i, frequencyIndex + 1] = 0
                if (frequencyIndex > 0):
                    remainingEnergy[i, frequencyIndex - 1] = 0

                i += 1

            noteEnd = i - 1 - k # go back to frame above threshold

            # backward pass
            i = noteCentre - 1
            k = 0
            
            # go backwards until start of audio or energy is below tolerance
            while (i > 0 and k < energyTolerance):
                if (remainingEnergy[i, frequencyIndex] < frameThreshold):
                    k += 1
                else:
                    k = 0

                # lock energy for pitch bin and neighbouring pitch bins
                remainingEnergy[i, frequencyIndex] = 0
                if (frequencyIndex < MAX_FREQUENCY_INDEX):
                    remainingEnergy[i, frequencyIndex + 1] = 0
                if (frequencyIndex > 0):
                    remainingEnergy[i, frequencyIndex - 1] = 0

                i -= 1

            noteStart = i + 1 + k # go back to frame above threshold
            
            if (noteEnd - noteStart <= minimumNoteLength): # skip if note is too short
                continue

            # add note
            amplitude = np.mean(frames[noteStart:noteEnd, frequencyIndex])
            noteEvents.append(
                (
                    noteStart,
                    noteEnd,
                    frequencyIndex + MIDI_OFFSET,
                    amplitude
                )
            )
    return noteEvents

# convert note creation output from frames to seconds
def framesToSeconds(noteEvents, sampleRate, hopSize):
    secondsPerFrame = hopSize / sampleRate

    converted = []
    for start, end, pitch, amplitude in noteEvents:
        startTime = start * secondsPerFrame
        endTime = end * secondsPerFrame

        converted.append((startTime, endTime, pitch, amplitude))

    return converted