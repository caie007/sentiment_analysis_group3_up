"""
05_run_robustness.py — retrain the best model on code-switch-augmented data (tag cs_aug),
every language x every seed; each run is evaluated on both clean and perturbed test sets.
Resumable like 04.
"""
import argparse, subprocess, sys
import pandas as pd
from config import *

p = argparse.ArgumentParser()
p.add_argument("--model", default=BEST_MODEL, choices=list(MODELS))
p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
p.add_argument("--langs", nargs="+", default=LANGS)
args = p.parse_args()
if not CS_TRAIN_CSV.exists():
    raise SystemExit("run 03_augment.py first")

def done():
    if not RUNS_CSV.exists():
        return set()
    d = pd.read_csv(RUNS_CSV)
    d = d[(d.tag == "cs_aug") & (d.test_set == "clean")]
    return set(zip(d.model, d.lang, d.seed))

trainer = ROOT / "scripts" / "02_train.py"
total = len(args.seeds) * len(args.langs); k = 0
for seed in args.seeds:
    for lang in args.langs:
        k += 1
        if (args.model, lang, seed) in done():
            print(f"[{k}/{total}] skip {lang} s{seed} (done)")
            continue
        print(f"\n[{k}/{total}] cs_aug | {args.model} | {lang} | seed {seed}", flush=True)
        subprocess.run([sys.executable, str(trainer), "--model", args.model, "--lang", lang,
                        "--seed", str(seed), "--tag", "cs_aug", "--train_csv", str(CS_TRAIN_CSV)], check=False)
print("robustness runs complete")
