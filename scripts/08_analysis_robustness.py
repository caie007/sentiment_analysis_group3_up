"""
08_analysis_robustness.py — RQ3.3 (robustness lead).
(1) Robustness to code-switching: clean-trained vs cs_aug-trained models on clean and perturbed tests.
(2) Calibration: expected calibration error (ECE, 15 bins) from saved probabilities, per model,
    plus reliability diagrams. Writes results/analysis/robustness_*.csv/.png and calibration_*.csv/.png
"""
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

d = pd.read_csv(RUNS_CSV)
pd.set_option("display.width", 250)

# ---------- (1) clean vs perturbed test, per training regime ----------
piv = d.groupby(["tag", "model", "lang", "test_set"]).f1_weighted.mean().unstack("test_set").reset_index()
has_cs = "cs" in piv.columns
if has_cs:
    piv["drop_clean_minus_cs"] = (piv["clean"] - piv["cs"]).round(2)
piv = piv.round(2)
piv.to_csv(ANALYSIS / "robustness_per_language.csv", index=False)
cols = [c for c in ["clean", "cs", "drop_clean_minus_cs"] if c in piv.columns]
summ = piv.groupby(["tag", "model"])[cols].mean().round(2)
summ.to_csv(ANALYSIS / "robustness_summary.csv")
print("Clean vs perturbed test, weighted F1 (mean over langs & seeds):\n", summ)

# ---------- (2) calibration / ECE ----------
def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; acc_b, conf_b, n_b = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            acc_b.append(np.nan); conf_b.append(np.nan); n_b.append(0); continue
        a, c = correct[m].mean(), conf[m].mean(); e += m.mean() * abs(a - c)
        acc_b.append(a); conf_b.append(c); n_b.append(int(m.sum()))
    return e, np.array(acc_b), np.array(conf_b), np.array(n_b)

rows, curves = [], {}
for f in sorted(PREDS.glob("*.csv")):
    # filename: <tag>__<model>__<lang>__s<seed>__<testset>.csv
    try:
        tag, model, lang, seed, testset = f.stem.split("__")
        seed = int(seed[1:])
    except ValueError:
        print(f"skipping unrecognised prediction file {f.name}"); continue
    p = pd.read_csv(f)
    probs = p[["p_neg", "p_neu", "p_pos"]].values
    conf = probs.max(1); correct = (p.pred == p.true).values.astype(float)
    e, ab, cb, nb = ece(conf, correct)
    rows.append({"tag": tag, "model": model, "lang": lang, "seed": seed, "test_set": testset,
                 "ece": round(100 * e, 2), "mean_conf": round(100 * conf.mean(), 2),
                 "accuracy": round(100 * correct.mean(), 2)})
    curves.setdefault((tag, model, testset), []).append((ab, cb, nb))
cal = pd.DataFrame(rows)
cal.to_csv(ANALYSIS / "calibration_per_run.csv", index=False)
cal_s = cal.groupby(["tag", "model", "test_set"])[["ece", "mean_conf", "accuracy"]].mean().round(2)
cal_s["overconfidence"] = (cal_s.mean_conf - cal_s.accuracy).round(2)
cal_s.to_csv(ANALYSIS / "calibration_summary.csv")
print("\nCalibration (ECE %, mean over langs & seeds; overconfidence = mean confidence - accuracy):\n", cal_s)
cal.groupby(["tag", "model", "test_set", "lang"]).ece.mean().round(2).unstack("lang").to_csv(ANALYSIS / "calibration_per_language.csv")

# ---------- figures ----------
if has_cs:
    keys = list(summ.index)
    fig, ax = plt.subplots(figsize=(11, 4.2)); x = np.arange(12); w = .8 / max(len(keys), 1)
    for k, (t, mdl) in enumerate(keys):
        g = piv[(piv.tag == t) & (piv.model == mdl)].set_index("lang").reindex(LANGS)
        ax.bar(x + (k - (len(keys) - 1) / 2) * w, g["drop_clean_minus_cs"], w, label=f"{mdl} ({t})")
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(LANGS)
    ax.set_ylabel("wF1 drop under code-switch perturbation")
    ax.set_title("RQ3.3 - robustness: clean-test F1 minus perturbed-test F1 (lower is better)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(ANALYSIS / "robustness_drop_per_language.png", dpi=200)

models = [mm for mm in MODELS if any(k[1] == mm for k in curves)]
fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), sharey=True)
for ax, mdl in zip(np.atleast_1d(axes), models):
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    for (t, mm, ts), cs in curves.items():
        if mm != mdl or ts != "clean":
            continue
        A = np.nanmean([c[0] for c in cs], 0); C = np.nanmean([c[1] for c in cs], 0)
        ax.plot(C, A, "o-", ms=3, label=f"{t} (ECE {cal_s.loc[(t, mdl, 'clean'), 'ece']:.1f})")
    ax.set_title(mdl); ax.set_xlabel("confidence"); ax.legend(fontsize=8); ax.grid(alpha=.3)
np.atleast_1d(axes)[0].set_ylabel("accuracy in bin")
fig.suptitle("Reliability diagrams (clean test, averaged over languages & seeds)")
fig.tight_layout(); fig.savefig(ANALYSIS / "calibration_reliability.png", dpi=200)
print(f"\nwritten to {ANALYSIS}")
