import mir_eval
from getMidiData import getNoteData, getIntervals, getPitchesInHZ, getPitchesInMIDI

estimatePath = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\FITP\output.mid')
estimateIntervals = getIntervals(estimatePath)
estimatePitches = getPitchesInHZ(estimatePath)

referencePath = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\FITP\FITPeval.mid')
referenceIntervals = getIntervals(referencePath)
referencePitches = getPitchesInHZ(referencePath)

print(estimateIntervals)
print(estimatePitches)

"""
- mir_eval uses hz for pitch
- use mir_eval.util.midi_to_hz to convert midi to pitch

- for intervals (onsets and offsets), mir_eval uses a 2d array to capture onset and offset (e.g. [onset, offset])
"""

mir_eval.transcription.validate(referenceIntervals, referencePitches, estimateIntervals, estimatePitches)
overlapScore = mir_eval.transcription.precision_recall_f1_overlap(referenceIntervals, referencePitches, estimateIntervals, estimatePitches)
melodyScore = mir_eval.melody.overall_accuracy
print("finished")
print("overlap score -> "+overlapScore)