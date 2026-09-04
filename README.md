# AfriSenti replication

Group replication of the monolingual baselines (Table 7) in Muhammad et al. (2023),
*AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages*.

Members: Kudzai (AfroXLMR-base), Vincent (XLM-R-base), Letsoba (AfriBERTa-large).

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

## Run
python scripts/01_load_data.py          # download data
python scripts/03_run_all.py            # train all models (~2 h on one GPU)
python scripts/04_summarise.py          # compare to the paper
python scripts/05_extra_experiments.py  # seed + epoch checks
python scripts/06_extra_summary.py

Results and figures are in `results/`. The full report is `AfriSenti_Replication_Report.docx`.