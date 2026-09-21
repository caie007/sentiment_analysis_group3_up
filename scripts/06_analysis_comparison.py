"""
06_analysis_comparison.py — RQ3.1 (comparison lead).
Aggregates clean runs over seeds, compares to the paper, and tests whether model differences
are significant across seeds (paired t-test + Wilcoxon over the 12 languages, and a
seed-level bootstrap CI on the mean difference). Writes results/analysis/comparison_*.csv/.png
"""
import itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from config import *

d = pd.read_csv(RUNS_CSV)
d = d[(d.tag == "clean") & (d.test_set == "clean")]
models = [m for m in MODELS if m in set(d.model)]
print(f"{len(d)} clean runs; seeds per model: {d.groupby('model').seed.nunique().to_dict()}")

# ---- per (model, lang): mean ± sd over seeds ----
g = d.groupby(["model", "lang"]).f1_weighted.agg(["mean", "std", "count"]).reset_index()
mean = g.pivot(index="lang", columns="model", values="mean").reindex(LANGS)
sd = g.pivot(index="lang", columns="model", values="std").reindex(LANGS).fillna(0)
out = pd.DataFrame(index=LANGS + ["AVG"])
for m in models:
    out[f"{m} mean"] = list(mean[m].round(1)) + [round(mean[m].mean(), 1)]
    out[f"{m} sd"] = list(sd[m].round(1)) + [round(sd[m].mean(), 1)]
    out[f"{m} paper"] = PAPER[m] + [round(float(np.mean(PAPER[m])), 1)]
    out[f"{m} delta"] = (out[f"{m} mean"] - out[f"{m} paper"]).round(1)
out.index.name = "lang"
out.to_csv(ANALYSIS / "comparison_table.csv")
pd.set_option("display.width", 250)
print("\n", out.to_string())

# ---- significance across seeds ----
per_seed = d.pivot_table(index=["lang", "seed"], columns="model", values="f1_weighted")
sig = []
for a, b in itertools.combinations(models, 2):
    x, y = mean[a].values, mean[b].values                 # 12 paired language means
    tt = stats.ttest_rel(x, y)
    try:
        wx_p = stats.wilcoxon(x, y).pvalue
    except ValueError:
        wx_p = float("nan")
    diffs = (per_seed[a] - per_seed[b]).dropna().values   # up to 36 (lang, seed) pairs
    rng = np.random.default_rng(0)
    boot = [rng.choice(diffs, len(diffs)).mean() for _ in range(5000)]
    sig.append({"A": a, "B": b, "mean_diff_A_minus_B": round(float(diffs.mean()), 2),
                "boot95_lo": round(float(np.percentile(boot, 2.5)), 2),
                "boot95_hi": round(float(np.percentile(boot, 97.5)), 2),
                "paired_t_p": round(float(tt.pvalue), 4), "wilcoxon_p": round(float(wx_p), 4),
                "A_wins_langs": int((x > y).sum()), "n_pairs": int(len(diffs))})
sig = pd.DataFrame(sig)
sig.to_csv(ANALYSIS / "comparison_significance.csv", index=False)
print("\nSignificance (paired over languages; bootstrap over lang x seed):\n", sig.to_string(index=False))

# ---- seed spread per model ----
spread = d.groupby(["model", "lang"]).f1_weighted.std().groupby("model").agg(["mean", "max"]).round(2)
spread.columns = ["mean_sd_over_seeds", "max_sd_over_seeds"]
spread.to_csv(ANALYSIS / "comparison_seed_spread.csv")
print("\nSeed spread:\n", spread)

# ---- figure ----
fig, ax = plt.subplots(figsize=(12, 4.6))
x = np.arange(12); n = len(models); w = .8 / n
for k, m in enumerate(models):
    off = (k - (n - 1) / 2) * w
    ax.bar(x + off, mean[m], w, yerr=sd[m], capsize=2, label=f"{m} (ours, mean±sd)")
    ax.plot(x + off, PAPER[m], "k_", ms=8, mew=1.5)
ax.plot([], [], "k_", ms=8, mew=1.5, label="paper (Table 7)")
ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_ylim(0, 95); ax.set_ylabel("weighted F1 (%)")
ax.set_title("RQ3.1 - clean fine-tuning, mean ± sd over seeds, vs. paper")
ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(ANALYSIS / "comparison_models_vs_paper.png", dpi=200)
print(f"\nwritten to {ANALYSIS}")
