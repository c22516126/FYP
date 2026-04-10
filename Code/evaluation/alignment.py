"""
todo list
- plot shift/f-score results
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from getMidiData import getPitchesInHZ, getIntervals, shiftIntervals
from src.config import AUDIO_SAMPLE_RATE, FFT_HOP
import mir_eval

# shift global timing by frame length from start to end range
def align(start, end, estimatePitches, estimateIntervals, referencePath):
    shiftSize = FFT_HOP / AUDIO_SAMPLE_RATE # shift by frame length (11.6ms)
    s = start - shiftSize

    referenceIntervals = getIntervals(referencePath)
    referencePitches = getPitchesInHZ(referencePath)
    fScorePeak = 0
    shifts = []
    
    peakShift = 0
    finalScore = [0, 0, 0, 0]
    while s < end:
        s += shiftSize
        shiftedEstimateIntervals = shiftIntervals(estimateIntervals, s)

        # offset is not considered

        precision, recall, f_measure, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
            referenceIntervals,
            referencePitches,
            shiftedEstimateIntervals,
            estimatePitches,
            offset_ratio=None
        )
        if f_measure > fScorePeak:
            peakShift = s
            fScorePeak = f_measure
            finalScore = [precision, recall, f_measure, avg_overlap_ratio]
        shifts.append(s)
    return shifts, peakShift, finalScore

# evaluate output pitch and interval lists against reference midi file
def evaluate(estimatePitches, estimateIntervals, reference):
    shifts, peakShift, finalScore = align(-0.2, 0.2, estimatePitches, estimateIntervals, reference)
    return shifts, peakShift, finalScore