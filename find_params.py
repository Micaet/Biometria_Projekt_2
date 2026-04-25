# tutaj eksperymentalnie znajduję optymalne parametry do
# klasyfikacji kodów


from itertools import combinations
import numpy as np
from app import BiometriaApp as ba
from utils import process_eye_projections_pro, unwrap_iris
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import os
from pathlib import Path

db_path = Path("train-subset")

data = {}  # data[person_id][side] = [path1, ..., path5]

for person_dir in sorted(db_path.iterdir()):
    if not person_dir.is_dir():
        continue
    person_id = person_dir.name
    data[person_id] = {"left": [], "right": []}

    for side in ["left", "right"]:
        side_dir = person_dir / side
        if not side_dir.exists():
            continue
        bmps = sorted(side_dir.glob("*.bmp"))
        data[person_id][side] = [str(p) for p in bmps]

for pid in list(data.keys())[:3]:
    print(pid, data[pid])


def extract_flat(path):
    result = process_eye_projections_pro(path)
    if result is None:
        return None
    _, _, _, _, flat = result
    return flat


genuine_scores = [] # porownanie oka tej samej osoby
# (z tej samej strony oczywiscie) (np. left/aeval1.bmp vs left/aeval5.bmp)
impostor_scores = [] # porownanie oka u roznych osob
# (tylko z tej samej strony, czyli lewe vs lewe)

def compute_all_scores(data, freq):
    genuine_scores = []
    impostor_scores = []
    persons = list(data.keys())

    flat_cache = {}
    for person_id in persons:
        for side in ["left", "right"]:
            paths = data[person_id][side]
            flats = [f for f in (extract_flat(p) for p in paths) if f is not None]
            flat_cache[(person_id, side)] = flats

    for side in ["left", "right"]:
        for person_id in persons:
            flats = flat_cache[(person_id, side)]  # ← z cache
            for f1, f2 in combinations(flats, 2):
                genuine_scores.append(ba.compare_iris(f1, f2, freq))

        for person_id in persons:
            flats_a = flat_cache[(person_id, side)]  # ← z cache
            if not flats_a:
                continue
            for other_id in persons:
                if other_id == person_id:
                    continue
                flats_b = flat_cache[(other_id, side)]  # ← z cache
                if flats_b:
                    impostor_scores.append(ba.compare_iris(flats_a[0], flats_b[0], freq))

    return genuine_scores, impostor_scores


results = []

for freq in np.linspace(0.05, 0.5, 80):

    genuine_scores, impostor_scores = compute_all_scores(data, freq)

    y_true = [1] * len(genuine_scores) + [0] * len(impostor_scores)
    y_score = [-s for s in genuine_scores + impostor_scores]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = -thresholds[eer_idx]

    results.append({
        'freq': freq,
        'auc': roc_auc, 'eer': eer,
        'threshold': eer_threshold
    })
    print(f"Freq: {freq}, AUC: {roc_auc}, EER: {eer}, threshold: {eer_threshold}")

best = max(results, key=lambda r: r['auc'])

persons = list(data.keys())
kf = KFold(n_splits=5, shuffle=True)

for train_idx, test_idx in kf.split(persons):
    train_persons = [persons[i] for i in train_idx]
    test_persons = [persons[i] for i in test_idx]

    g, imp = compute_all_scores(
        {p: data[p] for p in test_persons},
        best["freq"]
    )
    y_true = [1] * len(g) + [0] * len(imp)
    y_score = [-s for s in g + imp]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    print(f"Fold AUC: {auc(fpr, tpr):.4f}")


freqs = [r['freq'] for r in results]
aucs  = [r['auc']  for r in results]
eers  = [r['eer']  for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(freqs, aucs)
ax1.set_xlabel("freq")
ax1.set_ylabel("AUC")
ax1.set_title("AUC vs freq")
ax1.axvline(best['freq'], color='red', linestyle='--', label=f"best freq={best['freq']:.3f}")
ax1.legend()
ax1.grid(True)

ax2.plot(freqs, eers)
ax2.set_xlabel("freq")
ax2.set_ylabel("EER")
ax2.set_title("EER vs freq")
ax2.axvline(best['freq'], color='red', linestyle='--')
ax2.grid(True)

plt.tight_layout()
plt.savefig("freq_search.png", dpi=150)
plt.show()

g, imp = compute_all_scores(data, best['freq'])
y_true  = [1]*len(g) + [0]*len(imp)
y_score = [-s for s in g + imp]
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.scatter(fpr[eer_idx], tpr[eer_idx], color='red', zorder=5,
            label=f"EER = {eer:.4f}  (próg = {-thresholds[eer_idx]:.3f})")
plt.plot([0,1],[0,1],'k--', linewidth=0.8)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC — najlepsze freq={best['freq']:.3f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_best.png", dpi=150)
plt.show()

print(f"\nNajlepsze: freq={best['freq']:.4f}, AUC={best['auc']:.4f}, EER={best['eer']:.4f}, próg={best['threshold']:.4f}")