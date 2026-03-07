import mir_eval
from src.config import FITP_REFERENCE_PATH, FITP_ESTIMATE_PATH
from getMidiData import getNoteData, getIntervals, getPitchesInHZ, getPitchesInMIDI, getShiftedIntervals
from saveNotes import saveNotes
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

estimateIntervals = getIntervals(FITP_ESTIMATE_PATH)
estimatePitches = getPitchesInHZ(FITP_ESTIMATE_PATH)

referenceIntervals = getIntervals(FITP_REFERENCE_PATH)
referencePitches = getPitchesInHZ(FITP_REFERENCE_PATH)


"""
- mir_eval uses hz for pitch
- use mir_eval.util.midi_to_hz to convert midi to pitch

- for intervals (onsets and offsets), mir_eval uses a 2d array to capture onset and offset (e.g. [onset, offset])
"""

overlapScore = mir_eval.transcription.precision_recall_f1_overlap(referenceIntervals, referencePitches, estimateIntervals, estimatePitches)
overlapScore2 = mir_eval.transcription.precision_recall_f1_overlap(
    referenceIntervals, 
    referencePitches, 
    estimateIntervals, 
    estimatePitches,
    offset_ratio=None)  
    
print("finished")
print("overlap score -> "+str(overlapScore))
print("overlap score 2 -> "+str(overlapScore2))

savedRefNotes = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\notes files\refNotes.txt')
savedEstNotes = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\notes files\estNotes.txt')
saveNotes(FITP_ESTIMATE_PATH, savedEstNotes)
saveNotes(FITP_REFERENCE_PATH, savedRefNotes)


#audio_path = (r'C:\Users\jason\school\FYP\FYP\Code\input\FITPeval.mp3')
#model_output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
#midi_data.write("basic_pitch_output.mid")

#basicPitchPath = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\basic_pitch_output.mid')
#bpIntervals = getIntervals(basicPitchPath)
#bpPitches = getPitchesInHZ(basicPitchPath)

#bpScore = mir_eval.transcription.precision_recall_f1_overlap(
#    referenceIntervals,
#    referencePitches,
#    bpIntervals,
#    bpPitches
#)
#print("basic pitch score ->"+str(bpScore))
