"""
03_augment.py — build the code-switching augmentation (RQ3.3).

Training augmentation (label-preserving): for every training tweet in language L with label y,
splice a random 2-5-token span from a SAME-LABEL training tweet of a different language into a
random position. Output = original train rows + one augmented copy of each (2x training data).

Test perturbation (label-neutral noise): for every test tweet, splice a 2-5-token span from a
NEUTRAL-labelled training tweet of a different language. The donor carries no sentiment, so the
gold label is unchanged but the input now contains foreign-language material - a proxy for the
code-switching common on African Twitter. Both files are deterministic (seed 0).
"""
import random
import pandas as pd
from config import *

rng = random.Random(0)
df = pd.read_csv(BASE_CSV, dtype=str)
train = df[df.split == "train"].copy()

def splice(tweet, donor, rng):
    tw, dn = str(tweet).split(), str(donor).split()
    if len(dn) < 2:
        return tweet
    k = rng.randint(2, min(5, len(dn)))
    s = rng.randint(0, len(dn) - k)
    pos = rng.randint(0, len(tw))
    return " ".join(tw[:pos] + dn[s:s + k] + tw[pos:])

# pools: label -> {lang: [tweets]} from the training split
pool = {lab: {l: g.tweet.tolist() for l, g in grp.groupby("lang")} for lab, grp in train.groupby("label")}
def donor(lang, label, rng):
    other = [l for l in pool[label] if l != lang]
    return rng.choice(pool[label][rng.choice(other)])

# ---- training augmentation ----
aug_rows = [{"tweet": splice(r.tweet, donor(r.lang, r.label, rng), rng),
             "label": r.label, "lang": r.lang, "split": "train"}
            for r in train.itertuples(index=False)]
cs_train = pd.concat([train[["tweet", "label", "lang", "split"]], pd.DataFrame(aug_rows)], ignore_index=True)
cs_train.to_csv(CS_TRAIN_CSV, index=False)
print(f"augmented train: {len(train)} original + {len(aug_rows)} augmented -> {CS_TRAIN_CSV}")

# ---- test perturbation (neutral donors only) ----
test = df[df.split == "test"].copy()
test["tweet"] = [splice(r.tweet, donor(r.lang, "neutral", rng), rng) for r in test.itertuples(index=False)]
test.to_csv(CS_TEST_CSV, index=False)
print(f"perturbed test: {len(test)} rows -> {CS_TEST_CSV}")

for l in LANGS[:3]:
    n_orig = len(train[train.lang == l])
    ex = cs_train[cs_train.lang == l].iloc[n_orig]
    print(f"\n[{l} augmented example] {ex.tweet[:140]}")
