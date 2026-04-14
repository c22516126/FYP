import csv
import matplotlib.pyplot as plt
import os

def plotGridSearch(csv_path):
    params = []
    precision = []
    recall = []
    f1 = []
    overlap = []

    # extract param name from filename
    filename = os.path.basename(csv_path)
    param_name = filename.replace("Results.csv", "").replace(".csv", "")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            params.append(float(row[param_name]))
            precision.append(float(row["precision"]))
            recall.append(float(row["recall"]))
            f1.append(float(row["F1"]))
            overlap.append(float(row["overlap"]))

    # sort by parameter
    combined = list(zip(params, precision, recall, f1, overlap))
    combined.sort(key=lambda x: x[0])
    params, precision, recall, f1, overlap = zip(*combined)

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(params, precision, marker='o', label='Precision')
    plt.plot(params, recall, marker='o', label='Recall')
    plt.plot(params, f1, marker='o', label='F1')
    plt.plot(params, overlap, marker='o', label='Overlap')

    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'{param_name} Parameter Sweep')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(params)
    plt.tight_layout()

    plt.savefig(f"{param_name}Plot.png", dpi=150)
    plt.close()

def plotEvaluation(summary, outputPath="summaryPlot.png"):
    metrics = ["precision", "recall", "F1", "overlap"]
    values = [summary[m] for m in metrics]

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values)

    plt.ylabel("Score")
    plt.title("Final Evaluation Metrics")
    plt.ylim(0, 1)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(outputPath, dpi=150)
    plt.close()

def plotPerFileF1(metricsArray, outputPath="perFileF1.png"):
    f1_scores = metricsArray[:, 2]  # index 2 = F1

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(f1_scores)), f1_scores, marker='o')

    plt.xlabel("File Index")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per File")
    plt.ylim(0, 1)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(outputPath, dpi=150)
    plt.close()