# Sentiment Analysis in Code-Switched Social Media Text

University of Pretoria, COS 801 Deep Learning research project (2026), Group 3.
**Group members:** Gishon K Gwenzi (21731056), Letsoba Savannah Mabe (20582995), Vincent Mabuza (10633244)

---

## 1. What the project is about

Sentiment analysis of code-switched social media text remains a challenge,
especially for African languages with limited training data and representation
in multilingual language models. This study first reproduces the results of the
AfriSenti benchmark, then investigates the performance of Africa-centric
language models compared to a general-purpose multilingual model, investigates
the impact of language representation in pre-training, and finally evaluates
whether synthetic code-switching can improve classification robustness. Three
transformer-based models, AfroXLMR, AfriBERTa, and XLM-R, were fine-tuned and
evaluated on 12 African languages using the AfriSenti dataset. Model
performance was measured using weighted F1 scores across three random seeds.
The results show that AfroXLMR achieved the highest average weighted F1 score
of 63.9%, followed by AfriBERTa at 61.2% and XLM-R at 58.8%. Model performance
varied across languages, with Xitsonga, Moroccan Arabic, and Amharic being the
worst-performing languages for the respective models. These findings suggest
that training data availability and language representation during
pre-training influence classification performance. Inserting synthetic
code-switching into the test data reduced the performance of all three models
when trained on the original data. Retraining AfroXLMR with augmented data
improved its robustness, reducing the drop under code-switching from 2.2 to
1.5 F1 points, and improved its performance on both the original and
code-switched test data.

The project answers three research questions:

| | Question | Analysis script |
|---|---|---|
| RQ1 | Do Africa-centric encoders outperform a general-purpose encoder, and by how much per language? | `scripts/06_analysis_comparison.py` |
| RQ2 | How does performance vary across languages within the same model? | `scripts/07_analysis_fairness.py` |
| RQ3 | Can synthetic code-switching augmentation improve robustness without degrading performance on the original data? | `scripts/08_analysis_robustness.py` |

---

## 2. Files in this repository

### Top level

| File | What is in it |
|---|---|
| `README.md` | This file. |
| `requirements.txt` | The Python packages needed to run the code. |

### `scripts/` (run in the order of their numbers)

| File | What is in it |
|---|---|
| `scripts/config.py` | Shared settings used by every other script: folder paths, the 12 language codes and names, the three Hugging Face model ids, seeds (1, 2, 3), training settings (5 epochs, batch 16, learning rate 2e-5, max 128 tokens) and the AfriSenti paper's reference scores. |
| `scripts/00_eda.py` | Exploratory data analysis. Computes class balance, tweet length, vocabulary, unseen-word rate, duplicates, label shift between splits and script mix per language. Writes tables and figures to `results/eda/`. |
| `scripts/01_load_data.py` | Downloads the official AfriSenti train, dev and test files for the 12 languages and saves them as one file, `data/afrisenti_all.csv`. |
| `scripts/02_train.py` | Fine-tunes one encoder on one language with one seed, selects the best epoch on the dev set, evaluates on the original and the code-switched test sets, saves the predictions to `results/preds/` and adds the scores to `results/runs.csv`. |
| `scripts/03_augment.py` | Builds the synthetic code-switched data by span splicing (fixed seed 0): the augmented training set `data/afrisenti_cs_train.csv` and the perturbed test set `data/afrisenti_cs_test.csv`. |
| `scripts/04_run_grid.py` | Runs `02_train.py` for the full clean grid: 3 models x 12 languages x 3 seeds = 108 runs. Skips runs that are already recorded. |
| `scripts/05_run_robustness.py` | Retrains AfroXLMR on the augmented training set: 12 languages x 3 seeds = 36 runs. |
| `scripts/06_analysis_comparison.py` | RQ1. Builds the per-language comparison of the three models against the paper and runs the significance tests (paired t-test, Wilcoxon, bootstrap). |
| `scripts/07_analysis_fairness.py` | RQ2. Builds per-language weighted, macro and per-class F1 tables, and the best-versus-worst language summary. |
| `scripts/08_analysis_robustness.py` | RQ3. Compares scores on the original and code-switched test sets, with and without augmentation, and computes calibration error (ECE). |
| `scripts/09_baseline_tfidf.py` | Trains the TF-IDF baselines (logistic regression and linear SVM on word 1-2-grams and character 2-5-grams), tunes them on the dev set and compares them with the encoders. Writes to `results/baseline/`. |
| `scripts/run_overnight.py` | Runs every step above in sequence and logs progress to `results/overnight.log`. |

### `data/` (created by the scripts)

| File | What is in it |
|---|---|
| `data/afrisenti_all.csv` | All AfriSenti tweets in one table with columns `tweet`, `label`, `lang`, `split`. |
| `data/afrisenti_cs_train.csv` | Original training tweets plus one code-switched copy of each (same-label donor span). |
| `data/afrisenti_cs_test.csv` | Code-switched copy of all 34,307 test tweets (neutral donor span). |

### `results/`

| File | What is in it |
|---|---|
| `results/runs.csv` | One row per run and test set: model, language, seed, weighted F1, macro F1, accuracy, per-class F1 and training time. |
| `results/overnight.log` | Log of the full pipeline run, with the time taken by each stage. |
| `results/preds/` | One CSV per run and test set with the predicted label and class probabilities for every test tweet. |
| `results/eda/` | Output of `00_eda.py`: `eda_class_balance.csv`, `eda_length.csv`, `eda_vocabulary.csv`, `eda_duplicates.csv`, `eda_label_shift.csv`, `eda_script_mix.csv`, `eda_splits.csv`, `eda_surface.csv`, `eda_top_tokens.csv`, `eda_truncation.csv` and the matching figures. |
| `results/baseline/baseline_runs.csv` | One row per TF-IDF model, language and test set, with the selected settings and scores. |
| `results/baseline/baseline_summary.csv` | Average scores of the two TF-IDF models. |
| `results/baseline/baseline_vs_transformers.csv` | TF-IDF scores next to the three encoders, per language. |
| `results/baseline/baseline_vs_transformers.png` | Figure of the comparison above. |
| `results/baseline/preds/` | TF-IDF predictions per language and test set. |
| `results/analysis/comparison_table.csv` | Mean and standard deviation per language and model, next to the paper's scores. |
| `results/analysis/comparison_significance.csv` | Significance tests between each pair of models. |
| `results/analysis/comparison_seed_spread.csv` | Variation of each model across the three seeds. |
| `results/analysis/comparison_models_vs_paper.png` | Figure of our scores against the paper's. |
| `results/analysis/fairness_per_language_weighted.csv` | Weighted F1 per language and model. |
| `results/analysis/fairness_per_language_macro.csv` | Macro F1 per language and model. |
| `results/analysis/fairness_per_class.csv` | F1 for negative, neutral and positive per language and model. |
| `results/analysis/fairness_worst_group.csv` | Best and worst language, the gap between them and the correlation with training size, per model. |
| `results/analysis/fairness_heatmap_worstgroup.png` | Heatmap of scores per model and language. |
| `results/analysis/fairness_size_vs_f1.png` | Score against training-set size. |
| `results/analysis/robustness_summary.csv` | Average score on the original and code-switched test sets, with and without augmentation. |
| `results/analysis/robustness_per_language.csv` | The same comparison per language. |
| `results/analysis/robustness_drop_per_language.png` | Figure of the drop under code-switching per language. |
| `results/analysis/calibration_summary.csv` | Calibration error (ECE) per model. |
| `results/analysis/calibration_per_language.csv` | Calibration error per model and language. |
| `results/analysis/calibration_per_run.csv` | Calibration error per run. |
| `results/analysis/calibration_reliability.png` | Reliability diagram. |

---

## 3. Dataset

The project uses **AfriSenti** (Muhammad et al., 2023), the dataset of
SemEval-2023 Task 12: tweets in African languages labelled negative, neutral or
positive by native speakers.

| | |
|---|---|
| Where to find it | https://github.com/afrisenti-semeval/afrisent-semeval-2023 |
| Paper | https://aclanthology.org/2023.emnlp-main.862/ |
| Size | 111,718 tweets in 14 languages |
| Used here | The 12 languages with training data (106,828 tweets), with the official train, dev and test splits |
| Format | Tab-separated text files, one per language and split |

Training tweets per language: Hausa (14,172), Igbo (10,192), Yorùbá (8,522),
Amharic (5,984), Moroccan Arabic (5,583), Nigerian Pidgin (5,121), Twi (3,481),
Kinyarwanda (3,302), Mozambican Portuguese (3,063), Swahili (1,810), Algerian
Arabic (1,651) and Xitsonga (804). Oromo and Tigrinya have no training data and
are not used.

You do not need to download the dataset by hand: `scripts/01_load_data.py`
fetches the official files from the repository above. No text cleaning is
applied.

---

## 4. How to run the code

### Requirements

- Python 3.12
- A CUDA GPU for the encoders (tested with PyTorch 2.14, transformers 5.16 and
  datasets 5.0). The TF-IDF baselines and the analysis scripts run on a CPU.
- Internet access on the first run, to download the dataset and the three
  pretrained models.

### Set up

```bash
git clone https://github.com/caie007/sentiment_analysis_group3_up.git
cd sentiment_analysis_group3_up

python -m venv .venv
.venv\Scripts\activate            # Windows (on Linux/macOS: source .venv/bin/activate)

pip install -r requirements.txt
pip install sentencepiece protobuf    # needed by the AfriBERTa tokenizer
```

### Option A: run everything with one command

```bash
python scripts/run_overnight.py             # all steps, 3 seeds (about 9 hours on one GPU)
python scripts/run_overnight.py --seeds 1   # quicker single-seed pass (about 3.5 hours)
```

### Option B: run step by step

```bash
python scripts/01_load_data.py             # 1. download the dataset
python scripts/00_eda.py                   # 2. exploratory data analysis
python scripts/03_augment.py               # 3. build the code-switched train and test sets
python scripts/09_baseline_tfidf.py        # 4. TF-IDF baselines (about 5 minutes, CPU)
python scripts/04_run_grid.py              # 5. fine-tune 3 models x 12 languages x 3 seeds
python scripts/05_run_robustness.py        # 6. retrain AfroXLMR on augmented data
python scripts/06_analysis_comparison.py   # 7. RQ1 tables and tests
python scripts/07_analysis_fairness.py     # 8. RQ2 tables and figures
python scripts/08_analysis_robustness.py   # 9. RQ3 tables and figures
```

### Notes

- The pipeline can be stopped and restarted: runs already recorded in
  `results/runs.csv` are skipped.
- To reproduce only the tables and figures from the saved results, run steps
  7 to 9; they read `results/runs.csv` and `results/preds/` and need no GPU.
- All outputs are written to `results/`.

---

## Reference

Muhammad, S. H. et al. (2023). *AfriSenti: A Twitter Sentiment Analysis
Benchmark for African Languages.* EMNLP 2023.
