"""
todo list
- plot shift/f-score results
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from getMidiData import getPitchesInHZ, getIntervals, shiftIntervals
from src.config import AUDIO_SAMPLE_RATE, FFT_HOP, EVAL_FOLDER, CACHE_PATH
import mir_eval
import numpy as np
import time, pickle
from src.transcribe import transcribeWithParams, Transcriber
from src.pipeline.noteCreation import framesToSeconds
from getMidiData import noteToEvalData
from plot import plotEvaluation, plotPerFileF1
from getMidiData import buildDataset

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
    shifts, peakShift, finalScore = align(-0.3, 0, estimatePitches, estimateIntervals, reference)
    return shifts, peakShift, finalScore

# runs evaluation for each file within the dataset
def evaluateDataset(posteriorgrams, parameters):   
    metrics = []
    shiftValues = []

    for i, item in enumerate(posteriorgrams):
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
        _, peakShift, finalScore = evaluate(estimatePitches, estimatedIntervals, refPath)

        metrics.append(finalScore)
        shiftValues.append(peakShift)

    return {
        "metrics": np.array(metrics), 
        "shifts": np.array(shiftValues)
        }

def saveEvaluationResults(summary, outputPath="evalResults.csv"):
    import csv

    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "precision", "recall", "F1",
            "stdPrecision", "stdRecall", "stdF1",
            "overlap", "stdOverlap",
            "avgShift", "stdShift"
        ])

        writer.writerow([
            round(summary["precision"], 4),
            round(summary["recall"], 4),
            round(summary["F1"], 4),
            round(summary["stdPrecision"], 4),
            round(summary["stdRecall"], 4),
            round(summary["stdF1"], 4),
            round(summary["overlap"], 4),
            round(summary["stdOverlap"], 4),
            round(summary["avgShift"], 4),
            round(summary["stdShift"], 4)
        ])

    return outputPath

def savePerFileResults(metrics, shifts, outputPath="perFileResults.csv"):
    import csv

    with open(outputPath, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["fileIndex", "precision", "recall", "F1", "overlap", "shift"])

        for i in range(len(metrics)):
            writer.writerow([
                i,
                round(metrics[i][0], 4),
                round(metrics[i][1], 4),
                round(metrics[i][2], 4),
                round(metrics[i][3], 4),
                round(shifts[i], 4)
            ])

def runEvaluation(params):
    dataset = buildDataset(EVAL_FOLDER)

    # load model once
    start = time.time()
    transcriber = Transcriber()
    print(f"Model load: {time.time() - start:.2f} seconds")

    # load or compute posteriorgrams
    start = time.time()

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

    print(f"Inference loop: {time.time() - start:.2f} seconds")

    # run evaluation
    start = time.time()

    results = evaluateDataset(posteriorgrams, params)
    metrics = results["metrics"]
    shifts = results["shifts"]

    # summarise
    results = {
        "precision": np.mean(metrics[:, 0]),
        "recall": np.mean(metrics[:, 1]),
        "F1": np.mean(metrics[:, 2]),
        "overlap": np.mean(metrics[:, 3]),
        "avgShift": np.mean(shifts),

        "stdPrecision": np.std(metrics[:, 0]),
        "stdRecall": np.std(metrics[:, 1]),
        "stdF1": np.std(metrics[:, 2]),
        "stdOverlap": np.std(metrics[:, 3]),
        "stdShift": np.std(shifts)
    }

    print(f"Evaluation: {time.time() - start:.2f} seconds")

    # save + plot
    tag = paramsToString(params)

    evalPath = os.path.join("results", f"eval_{tag}.csv")
    perFilePath = os.path.join("results", f"perfile_{tag}.csv")
    evalPathPlot = os.path.join("results", f"eval_{tag}.png")
    perFilePathPlot = os.path.join("results", f"perfile_{tag}.png")
    
    os.makedirs("results", exist_ok=True)

    saveEvaluationResults(results, evalPath)
    savePerFileResults(metrics, shifts, perFilePath)

    plotEvaluation(results, evalPathPlot)
    plotPerFileF1(metrics, perFilePathPlot)

    return results

def paramsToString(params):
    return (
        f"on{params['onset']:.2f}_"
        f"fr{params['frame']:.2f}_"
        f"ml{params['min_len']}_"
        f"en{params['energy']}_"
        f"mel{int(params['melodia'])}"
    )
f1Focused = {
    "onset": 0.7,
    "frame": 0.3,
    "min_len": 11,
    "energy": 11,
    "melodia": False
}

pFocused = {
    "onset": 0.8,
    "frame": 0.4,
    "min_len": 14,
    "energy": 11,
    "melodia": False
}

rFocusedMelodiaOn = {
    "onset": 0.6,
    "frame": 0.3,
    "min_len": 8,
    "energy": 11,
    "melodia": True
}

rFocusedMelodiaOff = {
    "onset": 0.6,
    "frame": 0.3,
    "min_len": 8,
    "energy": 11,
    "melodia": False
}

if __name__ == "__main__":
    runEvaluation(rFocusedMelodiaOn)