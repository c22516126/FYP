"""
todo list
- plot shift/f-score results
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from getMidiData import getPitchesInHZ, getIntervals, getShiftedIntervals
from src.config import FITP_REFERENCE_PATH, FITP_ESTIMATE_PATH, AUDIO_SAMPLE_RATE, FFT_HOP
from src.config import CDL__ESTIMATE_PATH, CDL_REFERENCE_PATH
import mir_eval
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

# shift global timing by frame length from start to end range
def align(start, end, estimatePath, referencePath):
    shiftSize = FFT_HOP / AUDIO_SAMPLE_RATE # shift by frame length (11.6ms)
    s = start - shiftSize

    estimatePitches = getPitchesInHZ(estimatePath)
    estimateIntervals = getIntervals(estimatePath)
    referencePitches = getPitchesInHZ(referencePath)
    fScorePeak = 0
    scoreList = []

    while s < end:
        s += shiftSize
        shiftedReferenceIntervals = getShiftedIntervals(referencePath, s)

        # offset is not considered
        precision, recall, f_measure, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(shiftedReferenceIntervals, 
                                                                                                             referencePitches, 
                                                                                                             estimateIntervals, 
                                                                                                             estimatePitches, 
                                                                                                             offset_ratio=None)
        if f_measure > fScorePeak:
            peakShift = s
            fScorePeak = f_measure
            finalScore = [precision, recall, f_measure, avg_overlap_ratio]
        scoreList.append([s, f_measure])
    return peakShift, finalScore, scoreList
    

print("fitp")
# fish in the pool eval
pShift, fScore, sList = align(-0.2, 0.2, FITP_ESTIMATE_PATH, FITP_REFERENCE_PATH)
print(fScore)


print("clair de lune")

# claire de lune eval
pShift, fScore, sList = align(-0.2, 0.2, CDL__ESTIMATE_PATH, CDL_REFERENCE_PATH)
print(fScore)
