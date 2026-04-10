import sys
import os
import csv
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import numpy as np
from src.pipeline.noteCreation import createNotes, framesToSeconds
from src.transcribe import Transcriber
from getMidiData import noteToEvalData, buildDataset
from alignment import evaluate
from src.config import ONSET_DEFAULT, FRAME_DEFAULT, MIN_DEFAULT, ENERGY_DEFAULT, FFT_HOP, AUDIO_SAMPLE_RATE
from plot import plotCSVResults

EVAL_FOLDER = r'C:\Users\jason\school\FYP\FYP\Code\evaluation\midi files\eval'
CACHE_PATH = "posteriorgrams.pkl"

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
        melodia=False
    )

    return notes

def saveResults(results, testParameter):
    outputPath = f"{testParameter}Results.csv"

    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            testParameter,
            "precision", "recall", "F1",
            "stdPrecision", "stdRecall", "stdF1",
            "overlap", "stdOverlap",
            "avgShift", "stdShift"
        ])

        for r in results:
            paramValue = r[testParameter]
            if isinstance(paramValue, (int, float, np.integer, np.floating)):
                paramValue = round(paramValue, 3)

            writer.writerow([
                paramValue,
                round(r["precision"], 4),
                round(r["recall"], 4),
                round(r["F1"], 4),
                round(r["stdPrecision"], 4),
                round(r["stdRecall"], 4),
                round(r["stdF1"], 4),
                round(r["overlap"], 4),
                round(r["stdOverlap"], 4),
                round(r["avgShifts"], 4),
                round(r["stdShifts"], 4)
            ])

    return outputPath

# grid search

def gridSearch(testParameter, posteriorgrams, start, end, step):

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

    outputPath = saveResults(results, testParameter) # save to CSV
    plotCSVResults(outputPath)
    return results


dataset = buildDataset(EVAL_FOLDER)

start = time.time()

transcriber = Transcriber()

end = time.time()
print(f"Model load: {end - start:.2f} seconds")

start = time.time()
# precompute posteriorgrams

if os.path.exists(CACHE_PATH):
    print("Loading cached posteriorgrams")
    with open(CACHE_PATH, "rb") as f:
        posteriorgrams = pickle.load(f)

else:

    print("Running inference")
    posteriorgrams = []

    for audioPath, refPath in dataset:
        pitchFull, onsetFull, _ = transcriber._run_inference(audioPath)

        posteriorgrams.append({
            "pitch": pitchFull,
            "onset": onsetFull,
            "ref": refPath
        })

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(posteriorgrams, f)

    print("Saved posteriorgrams to cache")

end = time.time()

print(f"Inference loop: {end - start:.2f} seconds")

start = time.time()

results = gridSearch(
    testParameter = "energy",
    posteriorgrams = posteriorgrams,
    start = 0,
    end = 150,
    step = 30
)

end = time.time()

print(f"BP eval: {end - start:.2f} seconds")