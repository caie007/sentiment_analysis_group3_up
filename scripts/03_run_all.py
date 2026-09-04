"""
03_run_all.py — one model per group member, all 12 languages each.
Results accumulate in results/results.csv, stamped with the member's name.
"""
import subprocess, sys
from pathlib import Path

MEMBERS = {                       # name -> model they are responsible for
    "Kudzai":  "Davlan/afro-xlmr-base",
    "Vincent": "xlm-roberta-base",
    "Letsoba": "castorini/afriberta_large",
}
LANGS = ["amh", "arq", "ary", "hau", "ibo", "kin",
         "pcm", "por", "swa", "tso", "twi", "yor"]     # orm, tir have no train set
SKIP_DONE = {"Kudzai", "Vincent"} ###had fail issues
##SKIP_DONE = {"Kudzai"}            # already ran; remove from this set to re-run

script = Path(__file__).with_name("02_train_monolingual.py")
for member, model in MEMBERS.items():
    if member in SKIP_DONE:
        print(f"skipping {member} ({model}) — already done")
        continue
    for lang in LANGS:
        print(f"\n{'='*60}\n{member}  |  {model}  |  {lang}\n{'='*60}")
        subprocess.run([sys.executable, str(script),
                        "--lang", lang, "--model", model, "--member", member],
                       check=False)
print("\nAll done.")