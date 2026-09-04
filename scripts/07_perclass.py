"""
07_perclass.py — per-class test F1 for every saved prediction file.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score

RES = Path(__file__).resolve().parent.parent / "results"
LABELS = ["negative", "neutral", "positive"]

rows = []
for f in sorted(RES.glob("preds_*.csv")):
    d = pd.read_csv(f)
    per = f1_score(d.true, d.pred, labels=LABELS, average=None) * 100
    rows.append({"file": f.name, "neg": round(per[0], 1), "neu": round(per[1], 1), "pos": round(per[2], 1),
                 "never_predicted": ", ".join(l for l in LABELS if l not in set(d.pred)) or "-"})

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
out.to_csv(RES / "per_class_f1.csv", index=False)
print(f"\nsaved {RES / 'per_class_f1.csv'}")