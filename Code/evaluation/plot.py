import csv
import matplotlib.pyplot as plt
import os

def plotCSVResults(csv_path):
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

    # --- sort by parameter (IMPORTANT) ---
    combined = list(zip(params, precision, recall, f1, overlap))
    combined.sort(key=lambda x: x[0])
    params, precision, recall, f1, overlap = zip(*combined)

    # --- plot ---
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