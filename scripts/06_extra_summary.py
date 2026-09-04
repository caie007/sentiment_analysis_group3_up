"""
06_extra_summary.py — summarise the seed-variance and epoch-check runs.
"""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
df = pd.read_csv(RES / "results.csv")
PAPER = {"tso": 45.9, "twi": 62.6, "swa": 60.8, "amh": 54.9}   # AfroXLMR-base, Table 7

# ---- A) seed variance ----
base = df[(df.member == "Kudzai") & (df.model == "Davlan/afro-xlmr-base") & (df.epochs == 5)]
seeds = df[df.member == "Kudzai-seeds"]
allseeds = pd.concat([base[base.lang.isin(seeds.lang.unique())], seeds])

print("=== A) Seed variance — AfroXLMR-base, weighted F1 (%) ===")
rows = []
for lang, g in allseeds.groupby("lang"):
    f = g.f1_weighted * 100
    rows.append({"lang": lang, "n_seeds": len(f), "mean": round(f.mean(), 1),
                 "std": round(f.std(), 1), "min": round(f.min(), 1), "max": round(f.max(), 1),
                 "paper": PAPER[lang]})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(RES / "extra_seed_variance.csv", index=False)

# ---- B) epoch check ----
print("\n=== B) Epoch check — amh, AfroXLMR-base ===")
amh5 = base[base.lang == "amh"].f1_weighted.mean() * 100
amh10 = df[df.member == "Kudzai-epochs"].f1_weighted.mean() * 100
ep = pd.DataFrame([{"epochs": 5, "f1_weighted": round(amh5, 1), "paper": PAPER["amh"]},
                   {"epochs": 10, "f1_weighted": round(amh10, 1), "paper": PAPER["amh"]}])
print(ep.to_string(index=False))
ep.to_csv(RES / "extra_epoch_check.csv", index=False)