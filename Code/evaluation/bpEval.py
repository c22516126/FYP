from basic_pitch.inference import predict
import numpy as np
from alignment import evaluate
from gridSearch import saveResults, plotCSVResults, EVAL_FOLDER
from getMidiData import buildDataset

dataset = buildDataset(EVAL_FOLDER)

# basic note creation wrapper for comparion
def basicPitchTranscribe(audio_path):
    model_output, midi_data, note_events = predict(audio_path)

    # note_events = list of (start, end, pitch, confidence)
    
    intervals = []
    pitches = []

    for note in note_events:
        start, end, pitch, *_ = note
        intervals.append([start, end])
        pitches.append(pitch)

    return np.array(intervals), np.array(pitches)

def evaluateBasicPitch(dataset):
    metrics = []
    shiftValues = []

    for audioPath, refPath in dataset:
        estimatedIntervals, estimatePitches = basicPitchTranscribe(audioPath)

        if len(estimatedIntervals) == 0:
            metrics.append([0, 0, 0, 0])
            shiftValues.append(0)
            continue

        shifts, peakShift, finalScore = evaluate(
            estimatePitches,
            estimatedIntervals,
            refPath
        )

        metrics.append(finalScore)
        shiftValues.append(peakShift)

    metrics = np.array(metrics)
    shiftValues = np.array(shiftValues)

    # aggregate into same format as grid search
    result = {
        "model": "BasicPitch",
        "precision": np.mean(metrics[:, 0]),
        "recall": np.mean(metrics[:, 1]),
        "F1": np.mean(metrics[:, 2]),
        "stdPrecision": np.std(metrics[:, 0]),
        "stdRecall": np.std(metrics[:, 1]),
        "stdF1": np.std(metrics[:, 2]),
        "overlap": np.mean(metrics[:, 3]),
        "stdOverlap": np.std(metrics[:, 3]),
        "avgShifts": np.mean(shiftValues),
        "stdShifts": np.std(shiftValues)
    }

    results = [result]

    # reuse your existing pipeline
    outputPath = saveResults(results, "model")
    plotCSVResults(outputPath)

    return result

evaluateBasicPitch(dataset)