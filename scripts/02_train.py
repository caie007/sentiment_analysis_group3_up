"""
02_train.py — fine-tune one model on one language, one seed.
Evaluates on the clean test set and, if data/afrisenti_cs_test.csv exists, on the
code-switch-perturbed test set too. Saves predictions WITH probabilities (for ECE)
and appends one row per test set to results/runs.csv. Deletes the checkpoint afterwards.

  python scripts/02_train.py --model AfroXLMR-base --lang hau --seed 1
  python scripts/02_train.py --model AfroXLMR-base --lang hau --seed 1 --train_csv data/afrisenti_cs_train.csv --tag cs_aug
"""
import argparse, csv, shutil, time
import numpy as np, pandas as pd, torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from config import *

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=list(MODELS))
p.add_argument("--lang", required=True, choices=LANGS)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--epochs", type=int, default=EPOCHS)
p.add_argument("--batch", type=int, default=BATCH)
p.add_argument("--lr", type=float, default=LR)
p.add_argument("--train_csv", default=str(BASE_CSV))
p.add_argument("--tag", default="clean", help="run family: clean | cs_aug")
p.add_argument("--keep_ckpt", action="store_true")
args = p.parse_args()

hf_id = MODELS[args.model]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
run_id = f"{args.tag}__{args.model}__{args.lang}__s{args.seed}"
out_dir = ROOT / "models" / run_id

# ---------------- data ----------------
base = pd.read_csv(BASE_CSV, dtype=str)
trn = pd.read_csv(args.train_csv, dtype=str)

def part(df, split):
    d = df[(df.lang == args.lang) & (df.split == split)].copy()
    d["labels"] = d.label.map(label2id)
    return d[["tweet", "labels"]].reset_index(drop=True)

train_df, dev_df, test_df = part(trn, "train"), part(base, "dev"), part(base, "test")
if len(train_df) == 0:
    raise SystemExit(f"{args.lang}: no training data")
extra = {}
if CS_TEST_CSV.exists():
    extra["cs"] = part(pd.read_csv(CS_TEST_CSV, dtype=str), "test")
print(f"[{run_id}] train={len(train_df)} dev={len(dev_df)} test={len(test_df)} extra={list(extra)}")

tok = AutoTokenizer.from_pretrained(hf_id)
enc = lambda b: tok(b["tweet"], truncation=True, max_length=MAX_LEN)
to_ds = lambda d: Dataset.from_pandas(d, preserve_index=False).map(enc, batched=True, remove_columns=["tweet"])
ds = {"train": to_ds(train_df), "dev": to_ds(dev_df), "test": to_ds(test_df),
      **{k: to_ds(v) for k, v in extra.items()}}

# ---------------- model ----------------
torch.manual_seed(args.seed); np.random.seed(args.seed)
model = AutoModelForSequenceClassification.from_pretrained(
    hf_id, num_labels=3, id2label=id2label, label2id=label2id)

def metrics(pred):
    y, yh = pred.label_ids, pred.predictions.argmax(-1)
    return {"f1_weighted": f1_score(y, yh, average="weighted"),
            "f1_macro": f1_score(y, yh, average="macro"),
            "accuracy": accuracy_score(y, yh)}

targs = TrainingArguments(
    output_dir=str(out_dir), num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
    learning_rate=args.lr, warmup_steps=0.1, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="f1_weighted",
    fp16=torch.cuda.is_available(), logging_steps=100, report_to="none",
    seed=args.seed, disable_tqdm=True)
trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["dev"],
                  processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=metrics)

t0 = time.time()
trainer.train()
train_min = (time.time() - t0) / 60

# ---------------- evaluate on each test set ----------------
def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

rows = []
for name, d in [("clean", test_df)] + list(extra.items()):
    out = trainer.predict(ds["test" if name == "clean" else name])
    probs = softmax(out.predictions); y, yh = out.label_ids, probs.argmax(-1)
    per = f1_score(y, yh, labels=[0, 1, 2], average=None, zero_division=0)
    pd.DataFrame({"tweet": d.tweet.values,
                  "true": [id2label[i] for i in y], "pred": [id2label[i] for i in yh],
                  "p_neg": probs[:, 0], "p_neu": probs[:, 1], "p_pos": probs[:, 2]}
                 ).to_csv(PREDS / f"{run_id}__{name}.csv", index=False)
    row = {"tag": args.tag, "model": args.model, "lang": args.lang, "seed": args.seed, "test_set": name,
           "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
           "n_train": len(train_df), "n_test": len(d),
           "f1_weighted": round(100 * f1_score(y, yh, average="weighted"), 2),
           "f1_macro": round(100 * f1_score(y, yh, average="macro"), 2),
           "accuracy": round(100 * accuracy_score(y, yh), 2),
           "f1_neg": round(100 * per[0], 2), "f1_neu": round(100 * per[1], 2), "f1_pos": round(100 * per[2], 2),
           "train_min": round(train_min, 1)}
    rows.append(row)
    print(f"\n=== {run_id} | test={name} ===")
    print(classification_report(y, yh, target_names=LABELS, digits=4, zero_division=0))
    print(f"wF1 {row['f1_weighted']}  macroF1 {row['f1_macro']}  acc {row['accuracy']}  ({train_min:.1f} min)")

with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    if f.tell() == 0:
        w.writeheader()
    w.writerows(rows)
if not args.keep_ckpt:
    shutil.rmtree(out_dir, ignore_errors=True)
print(f"appended {len(rows)} row(s) to {RUNS_CSV}")
