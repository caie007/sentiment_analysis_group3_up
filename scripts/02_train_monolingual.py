"""
02_train_monolingual.py — Task A: fine-tune a multilingual model on ONE language.
Usage (Run > Edit Configurations > Parameters):
    --lang hau --model Davlan/afro-xlmr-base --member Kudzai
"""
import argparse, csv, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

# ---------------- arguments ----------------
p = argparse.ArgumentParser()
p.add_argument("--lang", default="tso")
p.add_argument("--model", default="Davlan/afro-xlmr-base")
p.add_argument("--member", default="unassigned")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
MODEL_TAG = args.model.split("/")[-1]
OUT_DIR = ROOT / "models" / f"{args.member}_{MODEL_TAG}_{args.lang}"

LABELS = ["negative", "neutral", "positive"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# ---------------- data ----------------
df = pd.read_csv(ROOT / "data" / "afrisenti_all.csv", dtype=str)
df = df[df["lang"] == args.lang].copy()
df["labels"] = df["label"].map(label2id)
splits = {s: df[df["split"] == s][["tweet", "labels"]] for s in ["train", "dev", "test"]}
if len(splits["train"]) == 0:
    raise SystemExit(f"{args.lang} has no training data (zero-shot language). Pick another.")
print(f"[{args.member}] {args.model} | {args.lang} |", {s: len(d) for s, d in splits.items()})

tok = AutoTokenizer.from_pretrained(args.model)
def encode(batch):
    return tok(batch["tweet"], truncation=True, max_length=args.max_len)
ds = {s: Dataset.from_pandas(d, preserve_index=False).map(encode, batched=True, remove_columns=["tweet"])
      for s, d in splits.items()}

# ---------------- model ----------------
torch.manual_seed(args.seed)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=3, id2label=id2label, label2id=label2id)

def metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    return {"f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred)}

targs = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=64,
    learning_rate=args.lr,
    warmup_steps=0.1,                 # float = ratio of total steps (transformers v5)
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=args.seed,
)

trainer = Trainer(
    model=model, args=targs,
    train_dataset=ds["train"], eval_dataset=ds["dev"],
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
)

# ---------------- train + test ----------------
t0 = time.time()
trainer.train()
train_min = (time.time() - t0) / 60

test_out = trainer.predict(ds["test"])
m = test_out.metrics
y_pred = test_out.predictions.argmax(-1)
print(f"\n=== {args.member} | {args.lang} | {args.model} | TEST ===")
print(classification_report(test_out.label_ids, y_pred, target_names=LABELS,
                            digits=4, zero_division=0))
print(f"Weighted F1: {m['test_f1_weighted']:.4f}   Macro F1: {m['test_f1_macro']:.4f}   "
      f"Acc: {m['test_accuracy']:.4f}   ({train_min:.1f} min)")

# ---------------- save predictions + summary row ----------------
pd.DataFrame({"tweet": splits["test"]["tweet"].values,
              "true": [id2label[i] for i in test_out.label_ids],
              "pred": [id2label[i] for i in y_pred]}
             ).to_csv(RESULTS / f"preds_{args.member}_{MODEL_TAG}_{args.lang}.csv", index=False)

row = {"member": args.member, "lang": args.lang, "model": args.model,
       "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed,
       "n_train": len(splits["train"]),
       "f1_weighted": round(m["test_f1_weighted"], 4),
       "f1_macro": round(m["test_f1_macro"], 4),
       "accuracy": round(m["test_accuracy"], 4),
       "train_min": round(train_min, 1)}
summary = RESULTS / "results.csv"
with open(summary, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=row.keys())
    if f.tell() == 0:
        w.writeheader()
    w.writerow(row)
print(f"Appended to {summary}")