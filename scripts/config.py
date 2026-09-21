"""Shared constants for the AfriSenti replication + analysis pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
ANALYSIS = RESULTS / "analysis"
for d in (DATA, RESULTS, PREDS, ANALYSIS):
    d.mkdir(parents=True, exist_ok=True)

BASE_CSV = DATA / "afrisenti_all.csv"
CS_TRAIN_CSV = DATA / "afrisenti_cs_train.csv"    # code-switch-augmented training data
CS_TEST_CSV = DATA / "afrisenti_cs_test.csv"      # code-switch-perturbed test data
RUNS_CSV = RESULTS / "runs.csv"                   # one row per (run, test set)

LANGS = ["amh", "arq", "ary", "hau", "ibo", "kin", "pcm", "por", "swa", "tso", "twi", "yor"]
LANG_NAMES = {"amh": "Amharic", "arq": "Algerian Arabic", "ary": "Moroccan Arabic", "hau": "Hausa",
              "ibo": "Igbo", "kin": "Kinyarwanda", "pcm": "Nigerian Pidgin", "por": "Mozambican Portuguese",
              "swa": "Swahili", "tso": "Xitsonga", "twi": "Twi", "yor": "Yorùbá"}
LABELS = ["negative", "neutral", "positive"]

MODELS = {                       # short name -> Hugging Face id
    "AfroXLMR-base":   "Davlan/afro-xlmr-base",
    "AfriBERTa-large": "castorini/afriberta_large",
    "XLM-R-base":      "xlm-roberta-base",
}
BEST_MODEL = "AfroXLMR-base"     # retrained with augmentation in the robustness stage
SEEDS = [1, 2, 3]
EPOCHS, BATCH, LR, MAX_LEN = 5, 16, 2e-5, 128

# Table 7 of Muhammad et al. (2023), weighted F1 (%), same language order as LANGS
PAPER = {
    "AfroXLMR-base":   [54.9, 65.5, 52.4, 77.2, 76.3, 67.2, 67.6, 66.6, 60.8, 45.9, 62.6, 70.0],
    "XLM-R-base":      [60.2, 65.9, 50.9, 73.2, 75.6, 56.7, 63.8, 70.1, 57.8, 47.4, 61.4, 62.7],
    "AfriBERTa-large": [56.9, 47.7, 44.1, 78.7, 78.6, 62.7, 62.3, 58.3, 61.5, 51.6, 65.2, 72.9],
}

LEADS = {"comparison": "RQ3.1 - aggregate results, paper comparison, significance across seeds",
         "fairness":   "RQ3.2 - per-language F1, worst-group analysis, per-class disparities",
         "robustness": "RQ3.3 - code-switching augmentation, perturbed-test robustness, calibration/ECE"}
