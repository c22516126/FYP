import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import mir_eval
from src.pipeline.noteCreation import createNotes, framesToSeconds
from src.transcribe import Transcriber
from getMidiData import noteToEvalData, getPitchesInHZ, getIntervals
from alignment import evaluate
from src.config import ONSET_DEFAULT, FRAME_DEFAULT, MIN_DEFAULT, ENERGY_DEFAULT, FFT_HOP, AUDIO_SAMPLE_RATE

# midi reference paths
HONEYMOON_RP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\123\123.mid')
CSM_RP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\CSM\CSM.mid')
UT_RP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\undertale\undertale.mid')
CDL_RP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\CDL\CDL.mid')
FITP_RP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\FITP\FITP.mid')

# audio paths
HONEYMOON_AP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\123\123.mp3')
CSM_AP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\CSM\CSM.mp3')
UT_AP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\undertale\undertale.mp3')
CDL_AP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\CDL\CDL.mp3')
FITP_AP = (r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\FITP\FITP.mp3')

# get pitch and interval arrays using tunable note creation params
def transcribeWithParams(pitchFull, onsetFull, params=None):
    if params is None:
        params = {
            "onset": 0.5,
            "frame": 0.3,
            "min_len": 11,
            "energy": 8
        }
    notes = createNotes(
        frames=pitchFull,
        onsets=onsetFull,
        onsetThreshold=params["onset"],
        frameThreshold=params["frame"],
        minimumNoteLength=params["min_len"],
        energyTolerance=params["energy"],
        melodia=True
    )

    return notes

# grid search

def gridSearch(testParameter, posteriorgrams, start, end, step):
    s = start - step

    results = []
    values = np.arange(start, end + step, step)

    for s in values:
        
        parameters = {
            "onset": ONSET_DEFAULT,
            "frame": FRAME_DEFAULT,
            "min_len": MIN_DEFAULT,
            "energy": ENERGY_DEFAULT
        }

        parameters[testParameter] = s
        metrics = []
        shiftValues = []

        # compute metrics for each midi file
        for item in posteriorgrams:
            pitchFull = item["pitch"]
            onsetFull = item["onset"]
            refPath = item["ref"]

            notes = transcribeWithParams(pitchFull, onsetFull, parameters)
            notesInSecs = framesToSeconds(notes, AUDIO_SAMPLE_RATE, FFT_HOP)

            estimatedIntervals, estimatePitches = noteToEvalData(notesInSecs)
            
            estimatedIntervals = np.array(estimatedIntervals).reshape(-1, 2)
            estimatePitches = np.array(estimatePitches)

            # onset only
            if len(estimatedIntervals) == 0:
                metrics.append([0, 0, 0, 0])
                shiftValues.append(0)
                continue
            shifts, peakShift, finalScore = evaluate(estimatePitches, estimatedIntervals, refPath)
            metrics.append(finalScore)
            shiftValues.append(peakShift)

        # get average metrics across dataset
        metrics = np.array(metrics)
        avgP = np.mean(metrics[:, 0])
        avgR = np.mean(metrics[:, 1])
        avgF1 = np.mean(metrics[:, 2])
        avgOverlap = np.mean(metrics[:, 3])
        avgShift = np.mean(shiftValues)

        # standard deviations across dataset
        stdP = np.std(metrics[:, 0])
        stdR = np.std(metrics[:, 1])
        stdF1 = np.std(metrics[:, 2])
        stdO = np.std(metrics[:, 3])
        stdShift = np.std(shiftValues)

        results.append({
            testParameter: s,
            "precision": avgP,
            "recall": avgR,
            "F1": avgF1,
            "stdPrecision": stdP,
            "stdRecall": stdR,
            "stdF1": stdF1,
            "overlap": avgOverlap,
            "stdOverlap": stdO,
            "avgShifts": avgShift,
            "stdShifts": stdShift
        })

    results = sorted(results, key=lambda x: x["F1"], reverse=True)

    output_path = f"{testParameter}Results.txt"

    with open(output_path, "w") as f:
        for r in results:
            f.write(
                f"{testParameter}={r[testParameter]:.3f} | "
                f"P={r['precision']:.3f} | "
                f"R={r['recall']:.3f} | "
                f"F1={r['F1']:.3f} | "
                f"shift={r['avgShifts']:.3f}\n"
            )
            
    return results

dataset = [
    (HONEYMOON_AP, HONEYMOON_RP),
    (CSM_AP, CSM_RP),
    (UT_AP, UT_RP),
    (CDL_AP, CDL_RP),
    (FITP_AP, FITP_RP),
]

transcriber = Transcriber()

# precompute posteriorgrams
posteriorgrams = []
for audioPath, refPath in dataset:
    pitchFull, onsetFull, _ = transcriber._run_inference(audioPath)

    posteriorgrams.append({
        "pitch": pitchFull,
        "onset": onsetFull,
        "ref": refPath
    })

results = gridSearch(
    testParameter = "frame",
    posteriorgrams = posteriorgrams,
    start = 0.3,
    end = 0.9,
    step = 0.1
)

