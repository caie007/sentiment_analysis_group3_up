"""
09_baseline_tfidf.py — classical TF-IDF baselines for every AfriSenti training language.

Features : word 1-2-grams + character 2-5-grams (within word boundaries), sublinear TF, min_df=2
Models   : TFIDF-LR   multinomial logistic regression (gives probabilities -> calibration)
           TFIDF-SVM  linear SVM
Selection: C in {0.1, 0.3, 1, 3, 10} x class_weight in {None, balanced}, chosen on dev weighted F1,
           then trained on train only and evaluated once on test — the same protocol as the
           transformer runs. Both models are deterministic, so one run per language is exact.
Tests    : clean test and, if data/afrisenti_cs_test.csv exists, the code-switch-perturbed test.

Writes to results/baseline/ (does NOT touch results/runs.csv, so 06-08 are unaffected):
  baseline_runs.csv               one row per model x language x test set
  baseline_vs_transformers.csv    per-language weighted & macro F1 against the three encoders
  baseline_summary.csv            12-language averages, gaps, robustness drop, ECE
  baseline_vs_transformers.png    grouped bars
  preds/                          per-tweet predictions (with probabilities for TFIDF-LR)

  python scripts/09_baseline_tfidf.py
"""
import time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
from config import *

warnings.filterwarnings("ignore")
OUT = RESULTS / "baseline"; BPREDS = OUT / "preds"
BPREDS.mkdir(parents=True, exist_ok=True)
GRID_C = [0.1, 0.3, 1, 3, 10]
GRID_W = [None, "balanced"]
pd.set_option("display.width", 220)

def make_features(train_text):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           lowercase=True, token_pattern=r"(?u)\b\w+\b")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           lowercase=True, max_features=200_000)
    word.fit(train_text); char.fit(train_text)
    return lambda texts: hstack([word.transform(texts), char.transform(texts)]).tocsr()

def build(name, C, w):
    if name == "TFIDF-LR":
        return LogisticRegression(C=C, class_weight=w, max_iter=3000)
    return LinearSVC(C=C, class_weight=w, max_iter=10000)

def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100 * e

base = pd.read_csv(BASE_CSV, dtype=str).fillna("")
cs = pd.read_csv(CS_TEST_CSV, dtype=str).fillna("") if CS_TEST_CSV.exists() else None
if cs is None:
    print("note: data/afrisenti_cs_test.csv not found — run 03_augment.py to add the perturbed test set")
lab2id = {l: i for i, l in enumerate(LABELS)}

rows = []
for lang in LANGS:
    part = lambda d, s: d[(d.lang == lang) & (d.split == s)]
    tr, dv, te = part(base, "train"), part(base, "dev"), part(base, "test")
    t0 = time.time()
    feats = make_features(tr.tweet)
    Xtr, Xdv, Xte = feats(tr.tweet), feats(dv.tweet), feats(te.tweet)
    ytr, ydv, yte = tr.label.map(lab2id).values, dv.label.map(lab2id).values, te.label.map(lab2id).values
    tests = {"clean": (te, Xte, yte)}
    if cs is not None:
        tc = part(cs, "test"); tests["cs"] = (tc, feats(tc.tweet), tc.label.map(lab2id).values)

    for name in ["TFIDF-LR", "TFIDF-SVM"]:
        best = None
        for C in GRID_C:
            for w in GRID_W:
                clf = build(name, C, w).fit(Xtr, ytr)
                f = f1_score(ydv, clf.predict(Xdv), average="weighted")
                if best is None or f > best[0]: best = (f, C, w)
        _, C, w = best
        clf = build(name, C, w).fit(Xtr, ytr)
        for ts, (d, X, y) in tests.items():
            yh = clf.predict(X)
            per = f1_score(y, yh, labels=[0, 1, 2], average=None, zero_division=0)
            row = {"model": name, "lang": lang, "test_set": ts, "C": C, "class_weight": w or "none",
                   "dev_f1_weighted": round(100 * best[0], 2), "n_train": len(tr), "n_test": len(d),
                   "n_features": X.shape[1],
                   "f1_weighted": round(100 * f1_score(y, yh, average="weighted"), 2),
                   "f1_macro": round(100 * f1_score(y, yh, average="macro"), 2),
                   "accuracy": round(100 * accuracy_score(y, yh), 2),
                   "f1_neg": round(100 * per[0], 2), "f1_neu": round(100 * per[1], 2), "f1_pos": round(100 * per[2], 2),
                   "ece": np.nan, "fit_sec": round(time.time() - t0, 1)}
            out = pd.DataFrame({"tweet": d.tweet.values, "true": [LABELS[i] for i in y], "pred": [LABELS[i] for i in yh]})
            if name == "TFIDF-LR":
                P = clf.predict_proba(X)
                out[["p_neg", "p_neu", "p_pos"]] = P
                row["ece"] = round(ece(P.max(1), (yh == y).astype(float)), 2)
            out.to_csv(BPREDS / f"{name}__{lang}__{ts}.csv", index=False)
            rows.append(row)
        r = rows[-len(tests)]
        print(f"{lang}  {name:9s}  C={C:<4} cw={str(w):8s}  dev {100 * best[0]:5.1f}  test wF1 {r['f1_weighted']:5.1f}  "
              f"macro {r['f1_macro']:5.1f}" + (f"  cs {rows[-1]['f1_weighted']:5.1f}" if cs is not None else ""))

runs = pd.DataFrame(rows)
runs.to_csv(OUT / "baseline_runs.csv", index=False)

# ------------------------------------------------------------------ compare with the encoders
clean = runs[runs.test_set == "clean"]
tab = clean.pivot(index="lang", columns="model", values="f1_weighted").reindex(LANGS)
tabm = clean.pivot(index="lang", columns="model", values="f1_macro").reindex(LANGS)
enc_models = []
if RUNS_CSV.exists():
    r = pd.read_csv(RUNS_CSV)
    r = r[(r.tag == "clean") & (r.test_set == "clean")]
    enc = r.groupby(["model", "lang"])[["f1_weighted", "f1_macro"]].mean()
    enc_models = [m for m in MODELS if m in enc.index.get_level_values(0)]
    for m in enc_models:
        tab[m] = enc.loc[m].f1_weighted.reindex(LANGS)
        tabm[m] = enc.loc[m].f1_macro.reindex(LANGS)
order = ["TFIDF-LR", "TFIDF-SVM"] + enc_models
comp = tab[order].round(1).add_suffix(" wF1").join(tabm[order].round(1).add_suffix(" macro"))
if enc_models:
    best_bl = tab[["TFIDF-LR", "TFIDF-SVM"]].max(axis=1)
    comp["best_baseline"] = tab[["TFIDF-LR", "TFIDF-SVM"]].idxmax(axis=1)
    comp["best_encoder"] = tab[enc_models].idxmax(axis=1)
    comp["gain_best_encoder_over_best_baseline"] = (tab[enc_models].max(axis=1) - best_bl).round(1)
comp.loc["AVG"] = comp.mean(numeric_only=True).round(1)
comp.to_csv(OUT / "baseline_vs_transformers.csv")
print("\nWeighted F1 (clean test): baselines vs encoders (encoders = mean over seeds)\n",
      comp[[c for c in comp.columns if c.endswith("wF1")] +
           ([ "gain_best_encoder_over_best_baseline"] if enc_models else [])].to_string())

summ = []
for name in ["TFIDF-LR", "TFIDF-SVM"]:
    g = runs[runs.model == name]
    c, p = g[g.test_set == "clean"], g[g.test_set == "cs"]
    summ.append({"model": name, "mean_wF1": round(c.f1_weighted.mean(), 1), "mean_macroF1": round(c.f1_macro.mean(), 1),
                 "worst_lang": c.loc[c.f1_weighted.idxmin(), "lang"], "worst_wF1": round(c.f1_weighted.min(), 1),
                 "near_zero_class_cells": int((c[["f1_neg", "f1_neu", "f1_pos"]] < 5).sum().sum()),
                 "langs_balanced_weights": int((c.class_weight == "balanced").sum()),
                 "mean_cs_wF1": round(p.f1_weighted.mean(), 1) if len(p) else np.nan,
                 "drop_clean_minus_cs": round(c.f1_weighted.mean() - p.f1_weighted.mean(), 2) if len(p) else np.nan,
                 "mean_ece": round(c.ece.mean(), 2) if c.ece.notna().any() else np.nan})
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "baseline_summary.csv", index=False)
print("\nBaseline summary:\n", summ.to_string(index=False))

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(12, 4.6)); x = np.arange(len(LANGS)); n = len(order); w = .8 / n
cols = ["#bdbdbd", "#7f7f7f", "#1f4e79", "#c55a11", "#548235"]
for k, m in enumerate(order):
    ax.bar(x + (k - (n - 1) / 2) * w, tab.loc[LANGS, m], w, label=m, color=cols[k % len(cols)])
ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_ylim(0, 90); ax.set_ylabel("weighted F1 (%)")
ax.set_title("TF-IDF baselines (grey) vs fine-tuned encoders (colour, mean over seeds), clean test")
ax.legend(fontsize=8, ncol=5, loc="upper left"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "baseline_vs_transformers.png", dpi=200)
print(f"\nwritten to {OUT}")
