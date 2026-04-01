import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alignment import align

from getMidiData import getPitchesInHZ, getIntervals, getShiftedIntervals
from src.config import FITP_RP, FITP_EP # fish in the pool
from src.config import CDL_EP, CDL_RP # claire de lune
import mir_eval
import mir_eval.display
import matplotlib.pyplot as plt
import numpy as np
#from basic_pitch import ICASSP_2022_MODEL_PATH
#from basic_pitch.inference import predict

# fish in the pool 
FITPpShift, FITPfScore, FITPshifts, FITPf1 = align(-0.2, 0.2, FITP_EP, FITP_RP)

# claire de lune 
CDLpShift, CDLfScore, CDLshifts, CDLf1 = align(-0.2, 0.2, CDL_EP, CDL_RP)

# CDL2
qPath = r'C:\Users\jason\school\FYP\FYP\Code\output\output_fs0_25.mid'
_,t1,_,_ = align(-0.2,0.2, qPath, CDL_RP)

qPath = r'C:\Users\jason\school\FYP\FYP\Code\output\output_fs0_5.mid'
_,t2,_,_ = align(-0.2,0.2, qPath, CDL_RP)

qPath = r'C:\Users\jason\school\FYP\FYP\Code\output\output_fs0_75.mid'
_,t3,_,_ = align(-0.2,0.2, qPath, CDL_RP)

qPath = r'C:\Users\jason\school\FYP\FYP\Code\output\output_fs1_0.mid'
_,t4,_,_ = align(-0.2,0.2, qPath, CDL_RP)

#print("Base system: "+str(CDLfScore))
print("Format: Precision, Recall, F1, Overlap Ratio")
print("0.25: "+str(t1))
print("0.5: "+str(t2))
print("0.75: "+str(t3))
print("1: "+str(t4))

# visualisation ------------------------------------------------------------------------------------------

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.plot(CDLshifts, CDLf1, label='Clair De Lune')  # Plot some data on the Axes.
ax.plot(FITPshifts, FITPf1, label='Fish in The Pool')

ax.axvline(FITPpShift, linestyle="--", alpha=0.5, color="tab:orange")
ax.axvline(CDLpShift, linestyle="--", alpha=0.5, color="tab:blue")

ax.set_title('Time Shift - F1 Relationship')
ax.set_ylabel("F-measure")
ax.set_xlabel("Time shift (seconds)")
ax.legend()

plt.savefig('timeShift.png')

# overlapping notes ------------------------------------------------------------------------------------------
# fish in the pool
plt.figure()
estIntervalsFITP = getIntervals(FITP_EP)
estPitchesFITP = getPitchesInHZ(FITP_EP)

refIntervalsFITP = getIntervals(FITP_RP)
refPitchesFITP = getPitchesInHZ(FITP_RP)
mir_eval.display.piano_roll(refIntervalsFITP, refPitchesFITP, color="tab:orange", label="Reference")
mir_eval.display.piano_roll(estIntervalsFITP, estPitchesFITP, color="tab:blue", alpha=0.5, label="Estimate")

plt.title("Estimated and Reference Transcription (Fish In The Pool)")
plt.xlabel("Time (seconds)")
plt.ylabel("Pitch (MIDI)")
plt.legend()
plt.savefig('FITPoverlappingNotes.png')

# clair de lune
plt.figure()
estIntervalsCDL = getIntervals(CDL_EP)
estPitchesCDL = getPitchesInHZ(CDL_EP)

refIntervalsCDL = getIntervals(CDL_RP)
refPitchesCDL = getPitchesInHZ(CDL_RP)
mir_eval.display.piano_roll(refIntervalsCDL, refPitchesCDL, color="tab:orange", label="Reference")
mir_eval.display.piano_roll(estIntervalsCDL, estPitchesCDL, color="tab:blue", alpha=0.5, label="Estimate")
plt.xlim(0, 60)
plt.title("Estimated and Reference Transcription (Clair De Lune)")
plt.xlabel("Time (seconds)")
plt.ylabel("Pitch (MIDI)")
plt.legend()
plt.savefig('CDLoverlappingNotes.png')

# Q factor experiment results
filter_scales = [0.25, 0.5, 0.75, 1.0]
results = [t1, t2, t3, t4]

precision =  [r[0] for r in results]
recall =     [r[1] for r in results]
f1 =         [r[2] for r in results]
overlap =    [r[3] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(filter_scales, precision, marker='o', label='Precision')
plt.plot(filter_scales, recall, marker='o', label='Recall')
plt.plot(filter_scales, f1, marker='o', label='F1')
plt.plot(filter_scales, overlap, marker='o', label='Overlap Ratio')
plt.xlabel('Q Factor Scale')
plt.ylabel('Score')
plt.title('Q-Factor Scaling (Clair de Lune)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(filter_scales)
plt.tight_layout()
plt.savefig('QFactor.png', dpi=150)

# bar chart
filter_scales = [0.25, 0.5, 0.75, 1.0]
results = [t1, t2, t3, t4]
metrics = ['Precision', 'Recall', 'F1', 'Overlap Ratio']

x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
for i, (fs, result) in enumerate(zip(filter_scales, results)):
    ax.bar(x + (i - 1.5) * width, result, width, label=fs)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Q-Factor Scaling (Clair de Lune)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('QFactorBar.png', dpi=150)