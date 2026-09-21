"""
04_run_grid.py — clean fine-tuning grid: every model x every language x every seed.
Resumable: skips any (model, lang, seed) already present in results/runs.csv.
"""
import argparse, subprocess, sys
import pandas as pd
from config import *

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=list(MODELS))
p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
p.add_argument("--langs", nargs="+", default=LANGS)
args = p.parse_args()

def done():
    if not RUNS_CSV.exists():
        return set()
    d = pd.read_csv(RUNS_CSV)
    d = d[(d.tag == "clean") & (d.test_set == "clean")]
    return set(zip(d.model, d.lang, d.seed))

trainer = ROOT / "scripts" / "02_train.py"
total = len(args.models) * len(args.seeds) * len(args.langs); k = 0
for model in args.models:
    for seed in args.seeds:
        for lang in args.langs:
            k += 1
            if (model, lang, seed) in done():
                print(f"[{k}/{total}] skip {model} {lang} s{seed} (done)")
                continue
            print(f"\n[{k}/{total}] {model} | {lang} | seed {seed}", flush=True)
            subprocess.run([sys.executable, str(trainer), "--model", model, "--lang", lang,
                            "--seed", str(seed), "--tag", "clean"], check=False)
print("grid complete")
