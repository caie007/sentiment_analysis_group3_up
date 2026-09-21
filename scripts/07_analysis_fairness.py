"""
07_analysis_fairness.py — RQ3.2 (fairness lead).
Per-language and per-class F1 for every model, worst-group (minimum-language) analysis,
best-worst gap, dispersion, and the relation between performance and training-set size.
Writes results/analysis/fairness_*.csv/.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from config import *

d = pd.read_csv(RUNS_CSV)
d = d[(d.tag == "clean") & (d.test_set == "clean")]
m = d.groupby(["model", "lang"])[["f1_weighted", "f1_macro", "f1_neg", "f1_neu", "f1_pos", "n_train"]].mean().reset_index()
models = [k for k in MODELS if k in set(m.model)]

# ---- per-language tables ----
wide = m.pivot(index="lang", columns="model", values="f1_weighted").reindex(LANGS)[models].round(1)
wide_m = m.pivot(index="lang", columns="model", values="f1_macro").reindex(LANGS)[models].round(1)
wide.to_csv(ANALYSIS / "fairness_per_language_weighted.csv")
wide_m.to_csv(ANALYSIS / "fairness_per_language_macro.csv")

# ---- worst-group summary per model ----
rows = []
for mod in models:
    g = m[m.model == mod].set_index("lang")
    cls = {c: (g[c].idxmin(), float(g[c].min())) for c in ["f1_neg", "f1_neu", "f1_pos"]}
    worst_c = min(cls, key=lambda c: cls[c][1])
    rho = stats.spearmanr(np.log(g.n_train), g.f1_weighted)
    rows.append({"model": mod,
                 "mean_wF1": round(g.f1_weighted.mean(), 1),
                 "worst_lang": g.f1_weighted.idxmin(), "worst_wF1": round(g.f1_weighted.min(), 1),
                 "best_lang": g.f1_weighted.idxmax(), "best_wF1": round(g.f1_weighted.max(), 1),
                 "best_worst_gap": round(g.f1_weighted.max() - g.f1_weighted.min(), 1),
                 "sd_across_langs": round(g.f1_weighted.std(), 1),
                 "langs_below_60": int((g.f1_weighted < 60).sum()),
                 "mean_macroF1": round(g.f1_macro.mean(), 1),
                 "worst_macro_lang": g.f1_macro.idxmin(), "worst_macroF1": round(g.f1_macro.min(), 1),
                 "worst_class_cell": f"{worst_c}@{cls[worst_c][0]}", "worst_class_F1": round(cls[worst_c][1], 1),
                 "near_zero_class_cells": int((g[["f1_neg", "f1_neu", "f1_pos"]] < 5).sum().sum()),
                 "spearman_rho_vs_logsize": round(float(rho.correlation), 2),
                 "spearman_p": round(float(rho.pvalue), 3)})
summ = pd.DataFrame(rows)
summ.to_csv(ANALYSIS / "fairness_worst_group.csv", index=False)
pd.set_option("display.width", 250)
print(summ.to_string(index=False))
print("\nPer-language weighted F1:\n", wide.to_string())

pc = m[["model", "lang", "f1_neg", "f1_neu", "f1_pos"]].round(1)
pc.to_csv(ANALYSIS / "fairness_per_class.csv", index=False)

# ---- figures ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2, 1]})
im = axes[0].imshow(wide.T.values, aspect="auto", cmap="viridis", vmin=30, vmax=85)
axes[0].set_xticks(range(12)); axes[0].set_xticklabels(LANGS)
axes[0].set_yticks(range(len(models))); axes[0].set_yticklabels(models)
for i_, mod in enumerate(models):
    for j, l in enumerate(LANGS):
        axes[0].text(j, i_, f"{wide.loc[l, mod]:.0f}", ha="center", va="center", fontsize=8, color="w")
axes[0].set_title("RQ3.2 - weighted F1 per language (mean over seeds)")
plt.colorbar(im, ax=axes[0], fraction=.02)
axes[1].bar(summ.model, summ.worst_wF1, color="#c55a11", label="worst language")
axes[1].bar(summ.model, summ.mean_wF1 - summ.worst_wF1, bottom=summ.worst_wF1, color="#1f4e79", alpha=.5, label="gap to mean")
for i_, r in summ.iterrows():
    axes[1].text(i_, r.worst_wF1 / 2, r.worst_lang, ha="center", fontsize=9, color="w", fontweight="bold")
axes[1].set_ylim(0, 80); axes[1].set_title("Worst-group F1 and gap to mean"); axes[1].legend(fontsize=8)
axes[1].tick_params(axis="x", rotation=15)
fig.tight_layout(); fig.savefig(ANALYSIS / "fairness_heatmap_worstgroup.png", dpi=200)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for mod in models:
    g = m[m.model == mod]
    ax.scatter(g.n_train, g.f1_weighted, label=mod)
    for l, n, f in zip(g.lang, g.n_train, g.f1_weighted):
        ax.annotate(l, (n, f), fontsize=7, xytext=(2, 2), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("training tweets (log scale)"); ax.set_ylabel("weighted F1 (%)")
ax.set_title("Performance vs. training-set size"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ANALYSIS / "fairness_size_vs_f1.png", dpi=200)
print(f"\nwritten to {ANALYSIS}")
