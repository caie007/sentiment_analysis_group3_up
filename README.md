# COS 801 Deep Learning Project: Sentiment Analysis in Code-Switched Social Media Text

University of Pretoria, COS 801 research project (2026), Group 3.
**Group members:** Gishon K Gwenzi (21731056), Letsoba Savannah Mabe (20582995), Vincent Mabuza (10633244)

Sentiment analysis of code-switched social media text remains a challenge, especially for African languages with limited training data and representation in multilingual language models. This study first reproduces the results of the Afrisenti benchmark, then investigates the performance of Africa-centric language models compared to a general-purpose multilingual model, investigates the impact of language representation in pre-training, and finally evaluates whether synthetic code-switching can improve classification robustness. Three transformer-based models, AfroXLMR, AfriBERTa, and XLM-R, were fine-tuned and evaluated on 12 African languages using the AfriSenti dataset. Model performance was measured using weighted F1 scores across three random seeds. The results show that AfroXLMR achieved the highest average weighted F1 score of 63.9%, followed by AfriBERTa at 61.2% and XLM-R at 58.8%. Model performance varied across languages, with Xitsonga, Moroccan Arabic, and Amharic being the worst-performing languages for the respective models. These findings suggest that training data availability and language representation during pre-training  influence classification performance. Synthetic code-switching reduced the overall performance of all three models when trained on the original data. Although synthetic code-switching initially reduced classification performance, retraining AfroXLMR with augmented data improved its robustness and performance from 2.2 to 1.5 F1 points on both the original and code-switched test data. 

| | |
|---|---|
| Exploratory data analysis | `scripts/00_eda.py` |
| RQ3.1 Model comparison (replication of the paper) | `scripts/06_analysis_comparison.py` |
| RQ3.2 Per-language fairness and worst-group analysis | `scripts/07_analysis_fairness.py` |
| RQ3.3 Code-switching robustness and calibration | `scripts/08_analysis_robustness.py` |
| TF-IDF baselines | `scripts/09_baseline_tfidf.py` |
| Whole pipeline in one command | `scripts/run_overnight.py` |

---

## Dataset

| | |
|---|---|
| Name | AfriSenti-SemEval 2023 (Task 12) |
| Publisher | Muhammad et al., 2023 |
| Paper | https://aclanthology.org/2023.emnlp-main.862/ |
| Repository | https://github.com/afrisenti-semeval/afrisent-semeval-2023 |
| File pattern | `data/<lang>/{train,dev,test}.tsv` in the repository above |
| Coverage | 111,718 tweets in 14 languages; 12 languages have training data (106,828 tweets) |
| Labels | negative, neutral, positive |
| Format | Tab-separated text (tweet, label) |
| Licence | See the AfriSenti repository; free for research use with citation of the paper |

**No tweets are committed to this repository.** `scripts/01_load_data.py`
downloads the official splits and writes `data/afrisenti_all.csv`.

### A note on loading the data

The Hugging Face dataset loaders for AfriSenti **do not work with
`datasets` 5.x** (script-based loaders were removed). This project reads the
TSV files directly from the AfriSenti GitHub repository instead, so the
official train / dev / test splits are used unchanged.

### Languages and splits

Defined in `scripts/config.py`. Training tweets per language:

| Size | Languages |
|---|---|
| Large | Hausa (14,172), Igbo (10,192), Yorùbá (8,522) |
| Medium | Amharic (5,984), Moroccan Arabic (5,583), Nigerian Pidgin (5,121), Twi (3,481), Kinyarwanda (3,302), Mozambican Portuguese (3,063) |
| Small | Swahili (1,810), Algerian Arabic (1,651), Xitsonga (804) |

Oromo and Tigrinya have dev and test data only and are not used.
No text cleaning is applied; tweets are used as published.

### Synthetic code-switched data

Built by `scripts/03_augment.py` with a fixed seed (0):

- **Augmented training set**  `data/afrisenti_cs_train.csv`. Each training
  tweet gets one copy with 2–5 tokens spliced in from a tweet in another
  language **with the same label** (2× training data).
- **Perturbed test set**  `data/afrisenti_cs_test.csv`. Each of the 34,307
  test tweets gets 2–5 tokens spliced in from a **neutral** tweet in another
  language, so the gold label is unchanged.

---

## Models

| Model | Hugging Face id | Pretraining |
|---|---|---|
| AfroXLMR-base | `Davlan/afro-xlmr-base` | XLM-R adapted to 17 African languages |
| AfriBERTa-large | `castorini/afriberta_large` | Trained from scratch on 11 African languages |
| XLM-R-base | `xlm-roberta-base` | 100 languages, general purpose |
| TF-IDF + LR / SVM | scikit-learn | None (word 1–2-grams and character 2–5-grams) |

Encoders: AdamW, learning rate 2e-5, batch 16, 5 epochs, 10% warm-up, weight
decay 0.01, max 128 tokens, best epoch chosen on dev weighted F1, seeds 1–3.
TF-IDF: `C` and class weights chosen on the dev set.

---

## Reproducing the analysis

Requires Python 3.12 and, for the encoders, a CUDA GPU (tested with
torch 2.14 + cu126, transformers 5.16, datasets 5.0). Internet access is needed
on the first run to download the data and the three models.

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; use source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install sentencepiece protobuf                  # needed by the AfriBERTa tokenizer

python scripts/run_overnight.py                     # everything, 3 seeds (~9 GPU-hours)
python scripts/run_overnight.py --seeds 1           # single-seed pass (~3.5 hours)
```

Or step by step:

```bash
python scripts/01_load_data.py           # download official splits
python scripts/00_eda.py                 # EDA tables and figures
python scripts/03_augment.py             # code-switched train and test sets
python scripts/09_baseline_tfidf.py      # TF-IDF baselines (~5 min, CPU)
python scripts/04_run_grid.py            # 3 models x 12 languages x 3 seeds
python scripts/05_run_robustness.py      # AfroXLMR retrained on augmented data
python scripts/06_analysis_comparison.py
python scripts/07_analysis_fairness.py
python scripts/08_analysis_robustness.py
```

The pipeline is resumable: runs already recorded in `results/runs.csv` are
skipped, so it is safe to re-run after an interruption. Progress is logged to
`results/overnight.log`.

---

## Layout

```
.
README.md
requirements.txt
scripts/               config, data loading, training, augmentation, analysis
data/                  created by 01_load_data.py and 03_augment.py
results/
 runs.csv              one row per (run, test set)
 overnight.log         pipeline log
 preds/                per-run test predictions with probabilities
 eda/                  EDA tables and figures
 baseline/             TF-IDF results and predictions
 analysis/             tables and figures for RQ3.1, RQ3.2, RQ3.3
```

## Outputs

| File | Contents |
|---|---|
| `results/runs.csv` | weighted F1, macro F1, accuracy, per-class F1 and training time for every run |
| `results/analysis/comparison_table.csv` | per-language mean and SD for each model against the paper's Table 7 |
| `results/analysis/comparison_significance.csv` | paired t-test, Wilcoxon and bootstrap intervals between models |
| `results/analysis/comparison_seed_spread.csv` | variation across seeds |
| `results/analysis/fairness_per_language_weighted.csv` | weighted F1 by language and model |
| `results/analysis/fairness_per_language_macro.csv` | macro F1 by language and model |
| `results/analysis/fairness_per_class.csv` | per-class F1 by language and model |
| `results/analysis/fairness_worst_group.csv` | best and worst language, gap, correlation with training size |
| `results/analysis/robustness_summary.csv` | clean vs. perturbed test scores, with and without augmentation |
| `results/analysis/robustness_per_language.csv` | perturbation drop by language |
| `results/analysis/calibration_summary.csv` | expected calibration error (15 bins) by model |
| `results/baseline/baseline_summary.csv` | TF-IDF LR and SVM headline results |
| `results/baseline/baseline_vs_transformers.csv` | TF-IDF against the three encoders by language |
| `results/eda/` | class balance, length, vocabulary, OOV, duplicates, label shift, script mix |

## Headline results

Mean weighted F1 over 12 languages (encoders: mean of 3 seeds):

| Model | Clean test | Perturbed test |
|---|---|---|
| AfroXLMR-base (paper: 63.9) | 63.9 | 61.7 |
| AfroXLMR-base + augmentation | 65.2 | 63.7 |
| TF-IDF + SVM | 63.5 | 62.2 |
| TF-IDF + LR | 63.3 | 62.3 |
| AfriBERTa-large (paper: 61.7) | 61.2 | 58.4 |
| XLM-R-base (paper: 62.1) | 58.8 | 56.3 |

XLM-R-base includes one diverged Amharic run (seed 2); its mean is 60.1
without it.

## Attribution

AfriSenti data © the AfriSenti authors. If you use this work, cite:

Muhammad, S. H. et al. (2023). *AfriSenti: A Twitter Sentiment Analysis
Benchmark for African Languages.* EMNLP 2023.

Models: AfroXLMR (Alabi et al., 2022), AfriBERTa (Ogueji et al., 2021),
XLM-R (Conneau et al., 2020).

## Report and presentation

Report: `report/` *(add the PDF here)*
Presentation: `presentation/COS_801_Project_Presentation.pptx` *(add the slides or a link here)*
