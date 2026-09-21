"""
01_load_data.py — download all 14 AfriSenti languages from the official GitHub repo
and save one combined CSV (data/afrisenti_all.csv).
"""
import pandas as pd
from config import *

BASE = "https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/data"
ALL14 = ["amh", "arq", "ary", "hau", "ibo", "kin", "orm", "pcm", "por", "swa", "tir", "tso", "twi", "yor"]
SPLITS = ["train", "dev", "test"]

frames = []
for lang in ALL14:
    counts = {}
    for split in SPLITS:
        url = f"{BASE}/{lang}/{split}.tsv"
        try:
            df = pd.read_csv(url, sep="\t", quoting=3, dtype=str)
        except pd.errors.EmptyDataError:      # orm / tir have no train set
            counts[split] = 0
            continue
        df = df[["tweet", "label"]].copy()
        df["lang"] = lang
        df["split"] = split
        frames.append(df)
        counts[split] = len(df)
    print(f"{lang}: {counts}")

all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv(BASE_CSV, index=False)
print("\nTotal rows:", len(all_df))
print(pd.crosstab(all_df["lang"], all_df["split"]))
print(all_df["label"].value_counts())
print(f"\nSaved to {BASE_CSV}")
