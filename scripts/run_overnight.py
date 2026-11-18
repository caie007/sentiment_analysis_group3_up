"""
run_overnight.py — the whole pipeline in sequence, logged to results/overnight.log.
Every training stage is resumable, so if the machine sleeps or a run crashes, just re-run this file.

  python scripts/run_overnight.py            # everything (~10 h on one consumer GPU)
  python scripts/run_overnight.py --seeds 1  # single-seed pass (~3.5 h)
"""
import argparse, subprocess, sys, time, datetime
from config import *

p = argparse.ArgumentParser()
p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
p.add_argument("--skip_grid", action="store_true")
p.add_argument("--skip_robust", action="store_true")
args = p.parse_args()
S = [str(s) for s in args.seeds]
SC = ROOT / "scripts"
LOG = RESULTS / "overnight.log"

def step(name, cmd, required=True):
    t0 = time.time(); stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'#' * 70}\n# {stamp}  {name}\n{'#' * 70}", flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{stamp} START {name}\n")
    r = subprocess.run([sys.executable, *map(str, cmd)], check=False)
    msg = f"{'OK' if r.returncode == 0 else 'FAIL'} {name} ({(time.time() - t0) / 60:.1f} min)"
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    if r.returncode and required:
        raise SystemExit(f"stopping: {name} failed")

if not BASE_CSV.exists():
    step("load data", [SC / "01_load_data.py"])
step("exploratory data analysis", [SC / "00_eda.py"], required=False)
step("build code-switch augmentation", [SC / "03_augment.py"])
step("TF-IDF baselines", [SC / "09_baseline_tfidf.py"], required=False)
if not args.skip_grid:
    step("clean grid (3 models x 12 langs x seeds)", [SC / "04_run_grid.py", "--seeds", *S])
if not args.skip_robust:
    step("cs_aug retraining of best model", [SC / "05_run_robustness.py", "--seeds", *S])
step("analysis: comparison (RQ3.1)", [SC / "06_analysis_comparison.py"], required=False)
step("analysis: fairness (RQ3.2)",   [SC / "07_analysis_fairness.py"],   required=False)
step("analysis: robustness (RQ3.3)", [SC / "08_analysis_robustness.py"], required=False)
print(f"\nAll done. Tables and figures in {ANALYSIS}; log in {LOG}")
