"""
04_summarise.py — compare results/results.csv against Table 7 of the AfriSenti paper.
Produces per-member comparison_<member>_<model>.csv/.png and one combined chart.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LANGS = ["amh", "arq", "ary", "hau", "ibo", "kin", "pcm", "por", "swa", "tso", "twi", "yor"]

# Table 7, Muhammad et al. 2023 (monolingual baselines, avg of 5 runs)
PAPER = {
    "Davlan/afro-xlmr-base":     [54.9, 65.5, 52.4, 77.2, 76.3, 67.2, 67.6, 66.6, 60.8, 45.9, 62.6, 70.0],
    "Davlan/afro-xlmr-large":    [61.6, 68.3, 56.6, 80.7, 79.5, 70.6, 68.7, 71.6, 63.4, 47.3, 64.3, 74.1],
    "xlm-roberta-base":          [60.2, 65.9, 50.9, 73.2, 75.6, 56.7, 63.8, 70.1, 57.8, 47.4, 61.4, 62.7],
    "castorini/afriberta_large": [56.9, 47.7, 44.1, 78.7, 78.6, 62.7, 62.3, 58.3, 61.5, 51.6, 65.2, 72.9],
}
MODEL2MEMBER = {"Davlan/afro-xlmr-base": "Kudzai",
                "xlm-roberta-base": "Vincent",
                "castorini/afriberta_large": "Letsoba"}

df = df[df.member.isin(["Kudzai", "Vincent", "Letsoba"])]

df = pd.read_csv(RES / "results.csv")
if "member" not in df.columns:
    df["member"] = "unassigned"
df["member"] = df["member"].fillna("unassigned")
df.loc[df.member == "unassigned", "member"] = df.model.map(MODEL2MEMBER)
print(f"{len(df)} runs in results.csv\n")

combined = {}
for (member, model), mdf in df.groupby(["member", "model"]):
    agg = mdf.groupby("lang")[["f1_weighted", "f1_macro", "accuracy"]].agg(["mean", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex(LANGS)

    out = pd.DataFrame({
        "lang": LANGS,
        "runs": agg["f1_weighted_count"].fillna(0).astype(int).values,
        "ours_f1w": (agg["f1_weighted_mean"] * 100).round(1).values,
        "ours_f1macro": (agg["f1_macro_mean"] * 100).round(1).values,
        "ours_acc": (agg["accuracy_mean"] * 100).round(1).values,
        "paper": PAPER.get(model, [None] * 12),
    })
    if model in PAPER:
        out["delta_f1w"] = (out["ours_f1w"] - out["paper"]).round(1)

    avg = {"lang": "AVG", "runs": int(out["runs"].sum()),
           "ours_f1w": round(out["ours_f1w"].mean(), 1),
           "ours_f1macro": round(out["ours_f1macro"].mean(), 1),
           "ours_acc": round(out["ours_acc"].mean(), 1)}
    if model in PAPER:
        avg["paper"] = round(sum(PAPER[model]) / 12, 1)
        avg["delta_f1w"] = round(avg["ours_f1w"] - avg["paper"], 1)
    out = pd.concat([out, pd.DataFrame([avg])], ignore_index=True)

    tag = f"{member}_{model.split('/')[-1]}"
    print(f"=== {member} — {model} ===")
    print(out.to_string(index=False), "\n")
    out.to_csv(RES / f"comparison_{tag}.csv", index=False)
    combined[f"{member} ({model.split('/')[-1]})"] = out[out.lang != "AVG"]["ours_f1w"].values

    body = out[out.lang != "AVG"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = range(len(body)); w = 0.38
    ax.bar([i - w/2 for i in x], body["ours_f1w"], w, label="Ours (weighted F1)")
    if model in PAPER:
        ax.bar([i + w/2 for i in x], body["paper"], w, label="Paper, Table 7")
    ax.set_xticks(list(x)); ax.set_xticklabels(body["lang"])
    ax.set_ylim(0, 100); ax.set_ylabel("score (%)")
    ax.set_title(f"{member} — {model}")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(RES / f"comparison_{tag}.png", dpi=200); plt.close(fig)

# one chart with all three models side by side
fig, ax = plt.subplots(figsize=(12, 5))
n = len(combined); w = 0.8 / n
for k, (name, vals) in enumerate(combined.items()):
    ax.bar([i + (k - (n-1)/2) * w for i in range(12)], vals, w, label=name)
ax.set_xticks(range(12)); ax.set_xticklabels(LANGS)
ax.set_ylim(0, 100); ax.set_ylabel("weighted F1 (%)")
ax.set_title("Monolingual baselines — all group members")
ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(RES / "comparison_all_models.png", dpi=200)
print("saved comparison_all_models.png")