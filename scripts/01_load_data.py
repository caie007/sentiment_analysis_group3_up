"""
01_load_data.py — Download all 14 AfriSenti languages from the official
GitHub repo and save one combined CSV.
"""
from pathlib import Path
import pandas as pd

BASE = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data"
LANGS = ["amh", "arq", "ary", "hau", "ibo", "kin", "orm",
         "pcm", "por", "swa", "tir", "tso", "twi", "yor"]
SPLITS = ["train", "dev", "test"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

frames = []
for lang in LANGS:
    counts = {}
    for split in SPLITS:
        url = f"{BASE}/{lang}/{split}.tsv"
        try:
            df = pd.read_csv(url, sep="\t", quoting=3, dtype=str)
        except pd.errors.EmptyDataError:      # orm / tir have no train set
            counts[split] = 0
            continue
        df = df[["tweet", "label"]]
        df["lang"] = lang
        df["split"] = split
        frames.append(df)
        counts[split] = len(df)
    print(f"{lang}: {counts}")

all_df = pd.concat(frames, ignore_index=True)
out = DATA_DIR / "afrisenti_all.csv"
all_df.to_csv(out, index=False)

print("\nTotal rows:", len(all_df))
print("\nRows per language and split:")
print(pd.crosstab(all_df["lang"], all_df["split"]))
print("\nLabel distribution:")
print(all_df["label"].value_counts())
print(f"\nSaved to {out}")