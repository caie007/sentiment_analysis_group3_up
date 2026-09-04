"""
05_extra_experiments.py — two robustness checks for the report.
  A) seed variance: 3 seeds on the three smallest languages (afro-xlmr-base)
  B) epoch check:   amh with the paper's 10 epochs instead of our 5
Total time ~30 min on GPU.
"""
import subprocess, sys
from pathlib import Path

script = Path(__file__).with_name("02_train_monolingual.py")
MODEL = "Davlan/afro-xlmr-base"

def run(*extra):
    subprocess.run([sys.executable, str(script), "--model", MODEL, *extra], check=False)

# A) seed variance
for lang in ["tso", "twi", "swa"]:
    for seed in [1, 2, 3]:
        print(f"\n{'='*60}\nSEED CHECK  |  {lang}  |  seed {seed}\n{'='*60}")
        run("--lang", lang, "--seed", str(seed), "--member", "Kudzai-seeds")

# B) epoch check
print(f"\n{'='*60}\nEPOCH CHECK  |  amh  |  10 epochs\n{'='*60}")
run("--lang", "amh", "--epochs", "10", "--member", "Kudzai-epochs")

print("\nExtra experiments done.")