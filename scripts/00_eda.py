"""
00_eda.py — exploratory data analysis of AfriSenti (the 12 languages with training data).

Reads data/afrisenti_all.csv and writes CSV tables + PNG figures to results/eda/:
  eda_splits.csv            split sizes per language
  eda_class_balance.csv     class counts/shares per language x split, imbalance ratio
  eda_label_shift.csv       train vs dev vs test label-distribution distance
  eda_length.csv            tweet length (characters, whitespace tokens) per language
  eda_truncation.csv        share of tweets longer than MAX_LEN subwords, per tokenizer
                            (skipped if the tokenizers can't be loaded)
  eda_script_mix.csv        Latin / Arabic / Ethiopic composition, mixed-script share
  eda_surface.csv           @user, hashtag, emoji, URL, digit, elongation rates
  eda_duplicates.csv        duplicate tweets within / across splits, label conflicts
  eda_vocabulary.csv        vocabulary size, type-token ratio, test OOV rate
  eda_top_tokens.csv        most class-indicative tokens per language (log-odds)
  eda_*.png                 five figures

  python scripts/00_eda.py
"""
import re, math
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

EDA = RESULTS / "eda"
EDA.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
C_NEG, C_NEU, C_POS = "#c0504d", "#9e9e9e", "#4f81bd"

df = pd.read_csv(BASE_CSV, dtype=str)
df = df[df.lang.isin(LANGS)].copy()
df["tweet"] = df.tweet.fillna("")
print(f"{len(df):,} tweets in the 12 training languages\n")

# ------------------------------------------------------------------ helpers
RE_USER = re.compile(r"@user", re.I)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_HASH = re.compile(r"#\w+")
RE_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF☀-➿⬀-⯿️]")
RE_DIGIT = re.compile(r"\d")
RE_ELONG = re.compile(r"(\w)\1{2,}")                       # 3+ repeated characters: "hhhh", "sooo"
RE_LATIN = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]")
RE_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
RE_ETHIOPIC = re.compile(r"[ሀ-᎟ⶀ-⷟꬀-꬯]")
RE_TOKEN = re.compile(r"\w+", re.UNICODE)

def tokens(text):
    return RE_TOKEN.findall(RE_USER.sub(" ", text).lower())

def normalise(text):                                       # for duplicate detection
    t = RE_USER.sub(" ", text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())

def dist(series):                                          # label distribution as a vector
    c = series.value_counts()
    return np.array([c.get(l, 0) for l in LABELS], dtype=float) / max(len(series), 1)

def jsd(p, q):                                             # Jensen-Shannon distance (base 2), 0..1
    m = (p + q) / 2
    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    return math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))

# ------------------------------------------------------------------ 1. splits and class balance
splits = df.groupby(["lang", "split"]).size().unstack().reindex(LANGS)[["train", "dev", "test"]]
splits["total"] = splits.sum(axis=1)
splits.to_csv(EDA / "eda_splits.csv")

rows = []
for l in LANGS:
    for s in ["train", "dev", "test"]:
        d = df[(df.lang == l) & (df.split == s)]
        c = d.label.value_counts()
        n = [int(c.get(x, 0)) for x in LABELS]
        rows.append({"lang": l, "split": s, "n": len(d),
                     **{f"n_{x}": v for x, v in zip(LABELS, n)},
                     **{f"share_{x}": round(100 * v / len(d), 1) for x, v in zip(LABELS, n)},
                     "majority_class": LABELS[int(np.argmax(n))], "minority_class": LABELS[int(np.argmin(n))],
                     "imbalance_ratio": round(max(n) / max(min(n), 1), 2),
                     "majority_baseline_acc": round(100 * max(n) / len(d), 1)})
bal = pd.DataFrame(rows)
bal.to_csv(EDA / "eda_class_balance.csv", index=False)

shift = []
for l in LANGS:
    p = {s: dist(df[(df.lang == l) & (df.split == s)].label) for s in ["train", "dev", "test"]}
    shift.append({"lang": l, "jsd_train_dev": round(jsd(p["train"], p["dev"]), 3),
                  "jsd_train_test": round(jsd(p["train"], p["test"]), 3),
                  **{f"train_{x}": round(100 * p["train"][k], 1) for k, x in enumerate(LABELS)},
                  **{f"test_{x}": round(100 * p["test"][k], 1) for k, x in enumerate(LABELS)}})
shift = pd.DataFrame(shift)
shift.to_csv(EDA / "eda_label_shift.csv", index=False)
print("Class balance (train) and train->test label shift:")
print(bal[bal.split == "train"].set_index("lang")[["n", "share_negative", "share_neutral", "share_positive",
      "minority_class", "imbalance_ratio"]].join(shift.set_index("lang")[["jsd_train_test"]]).to_string(), "\n")

# ------------------------------------------------------------------ 2. length
df["n_chars"] = df.tweet.str.len()
df["n_words"] = df.tweet.str.split().str.len()
length = df.groupby("lang").agg(chars_mean=("n_chars", "mean"), chars_median=("n_chars", "median"),
                                chars_p95=("n_chars", lambda x: x.quantile(.95)),
                                words_mean=("n_words", "mean"), words_median=("n_words", "median"),
                                words_p95=("n_words", lambda x: x.quantile(.95)), words_max=("n_words", "max")
                                ).reindex(LANGS).round(1)
by_label = df.groupby(["lang", "label"]).n_words.mean().unstack()[LABELS].reindex(LANGS).round(1)
by_label.columns = [f"words_mean_{c}" for c in by_label.columns]
length = length.join(by_label)
length.to_csv(EDA / "eda_length.csv")
print("Tweet length:\n", length[["chars_mean", "words_mean", "words_p95", "words_max"]].to_string(), "\n")

# ------------------------------------------------------------------ 3. subword truncation (needs cached tokenizers)
trunc_rows = []
try:
    from transformers import AutoTokenizer
    sample = pd.concat([g.sample(min(len(g), 2000), random_state=0) for _, g in df.groupby("lang")])
    for short, hf in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(hf)
        except Exception as e:
            print(f"  [truncation] could not load {hf}: {type(e).__name__} — skipped"); continue
        lens = [len(x) for x in tok(sample.tweet.tolist(), add_special_tokens=True)["input_ids"]]
        sample[f"sub_{short}"] = lens
        for l, g in sample.groupby("lang"):
            trunc_rows.append({"model": short, "lang": l, "subwords_mean": round(g[f"sub_{short}"].mean(), 1),
                               "subwords_p95": round(g[f"sub_{short}"].quantile(.95), 1),
                               "subwords_per_word": round((g[f"sub_{short}"] - 2).sum() / max(g.n_words.sum(), 1), 2),
                               f"pct_over_{MAX_LEN}": round(100 * (g[f"sub_{short}"] > MAX_LEN).mean(), 2)})
except ImportError:
    print("  [truncation] transformers not installed — skipped")
if trunc_rows:
    trunc = pd.DataFrame(trunc_rows)
    trunc.to_csv(EDA / "eda_truncation.csv", index=False)
    print("Subword fertility and truncation:\n",
          trunc.pivot(index="lang", columns="model", values="subwords_per_word").reindex(LANGS).to_string(), "\n",
          trunc.pivot(index="lang", columns="model", values=f"pct_over_{MAX_LEN}").reindex(LANGS).to_string(), "\n")

# ------------------------------------------------------------------ 4. script composition
def script_counts(text):
    t = RE_USER.sub(" ", RE_URL.sub(" ", text))
    return len(RE_LATIN.findall(t)), len(RE_ARABIC.findall(t)), len(RE_ETHIOPIC.findall(t))

sc = np.array([script_counts(t) for t in df.tweet])
df["n_latin"], df["n_arabic"], df["n_ethiopic"] = sc[:, 0], sc[:, 1], sc[:, 2]
letters = sc.sum(1)
df["n_letters"] = letters
# a script "counts" in a tweet if it contributes >= 3 letters and >= 10% of the tweet's letters
present = (sc >= 3) & (sc >= 0.10 * letters[:, None].clip(min=1))
df["n_scripts"] = present.sum(1)
df["dominant_script"] = np.where(letters == 0, "none", np.array(["Latin", "Arabic", "Ethiopic"])[sc.argmax(1)])
mix = df.groupby("lang").apply(lambda g: pd.Series({
    "pct_latin_letters": 100 * g.n_latin.sum() / max(g.n_letters.sum(), 1),
    "pct_arabic_letters": 100 * g.n_arabic.sum() / max(g.n_letters.sum(), 1),
    "pct_ethiopic_letters": 100 * g.n_ethiopic.sum() / max(g.n_letters.sum(), 1),
    "pct_tweets_mixed_script": 100 * (g.n_scripts >= 2).mean(),
    "pct_tweets_no_letters": 100 * (g.n_letters == 0).mean(),
    "dominant_script": g.dominant_script.mode().iat[0]}), include_groups=False).reindex(LANGS)
mix = mix.round({c: 1 for c in mix.columns if c.startswith("pct")})
mix.to_csv(EDA / "eda_script_mix.csv")
print("Script composition:\n", mix.to_string(), "\n")

# ------------------------------------------------------------------ 5. surface features
df["has_user"] = df.tweet.str.contains(RE_USER)
df["has_url"] = df.tweet.str.contains(RE_URL)
df["has_hashtag"] = df.tweet.str.contains(RE_HASH)
df["n_emoji"] = df.tweet.str.count(RE_EMOJI)
df["has_digit"] = df.tweet.str.contains(RE_DIGIT)
df["has_elongation"] = df.tweet.map(lambda t: bool(RE_ELONG.search(t)))
surf = df.groupby("lang").agg(pct_user=("has_user", "mean"), pct_url=("has_url", "mean"),
                              pct_hashtag=("has_hashtag", "mean"), pct_emoji=("n_emoji", lambda x: (x > 0).mean()),
                              emoji_per_tweet=("n_emoji", "mean"), pct_digit=("has_digit", "mean"),
                              pct_elongation=("has_elongation", "mean")).reindex(LANGS)
for c in surf.columns:
    if c.startswith("pct"): surf[c] = 100 * surf[c]
surf = surf.round(2)
emo_lab = df.groupby(["lang", "label"]).n_emoji.apply(lambda x: 100 * (x > 0).mean()).unstack()[LABELS].reindex(LANGS).round(1)
emo_lab.columns = [f"pct_emoji_{c}" for c in emo_lab.columns]
surf = surf.join(emo_lab)
surf.to_csv(EDA / "eda_surface.csv")
print("Surface features (% of tweets):\n", surf.to_string(), "\n")

# ------------------------------------------------------------------ 6. duplicates and leakage
df["norm"] = df.tweet.map(normalise)
dup_rows = []
for l in LANGS:
    d = df[(df.lang == l) & (df.norm.str.len() > 0)]
    row = {"lang": l}
    for s in ["train", "dev", "test"]:
        x = d[d.split == s]
        row[f"pct_dup_within_{s}"] = round(100 * x.norm.duplicated(keep=False).mean(), 2)
    tr = set(d[d.split == "train"].norm)
    for s in ["dev", "test"]:
        x = d[d.split == s]
        row[f"pct_{s}_in_train"] = round(100 * x.norm.isin(tr).mean(), 2) if len(tr) else 0.0
    g = d.groupby("norm").label.nunique()
    row["n_texts_with_conflicting_labels"] = int((g > 1).sum())
    row["pct_texts_with_conflicting_labels"] = round(100 * (g > 1).mean(), 2)
    dup_rows.append(row)
dups = pd.DataFrame(dup_rows)
cross = df[df.norm.str.len() > 0].groupby("norm").lang.nunique()
dups.to_csv(EDA / "eda_duplicates.csv", index=False)
print("Duplicates / leakage:\n", dups.to_string(index=False))
print(f"Normalised texts appearing in more than one language: {int((cross > 1).sum())}\n")

# ------------------------------------------------------------------ 7. vocabulary and OOV
voc_rows, top_rows = [], []
for l in LANGS:
    tr = df[(df.lang == l) & (df.split == "train")]
    te = df[(df.lang == l) & (df.split == "test")]
    tr_tok = [t for x in tr.tweet for t in tokens(x)]
    te_tok = [t for x in te.tweet for t in tokens(x)]
    c = Counter(tr_tok); vocab = set(c)
    voc_rows.append({"lang": l, "train_tokens": len(tr_tok), "train_types": len(vocab),
                     "type_token_ratio": round(len(vocab) / max(len(tr_tok), 1), 3),
                     "pct_hapax_types": round(100 * sum(1 for v in c.values() if v == 1) / max(len(vocab), 1), 1),
                     "test_oov_token_rate": round(100 * sum(t not in vocab for t in te_tok) / max(len(te_tok), 1), 1),
                     "test_oov_type_rate": round(100 * len(set(te_tok) - vocab) / max(len(set(te_tok)), 1), 1)})
    # class-indicative tokens: log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008)
    cls = {lab: Counter(t for x in tr[tr.label == lab].tweet for t in tokens(x)) for lab in LABELS}
    tot = sum(c.values()); a0 = 0.01 * tot
    for lab in LABELS:
        yi, rest = cls[lab], c - cls[lab]
        ni, nj = sum(yi.values()), sum(rest.values())
        scores = []
        for w, fw in c.items():
            if fw < 5: continue
            aw = 0.01 * fw
            d_ = math.log((yi[w] + aw) / (ni + a0 - yi[w] - aw)) - math.log((rest[w] + aw) / (nj + a0 - rest[w] - aw))
            var = 1 / (yi[w] + aw) + 1 / (rest[w] + aw)
            scores.append((d_ / math.sqrt(var), w, yi[w]))
        for rank, (z, w, f) in enumerate(sorted(scores, reverse=True)[:15], 1):
            top_rows.append({"lang": l, "label": lab, "rank": rank, "token": w, "z_score": round(z, 2), "count_in_class": f})
voc = pd.DataFrame(voc_rows)
voc.to_csv(EDA / "eda_vocabulary.csv", index=False)
top = pd.DataFrame(top_rows)
top.to_csv(EDA / "eda_top_tokens.csv", index=False)
print("Vocabulary:\n", voc.to_string(index=False), "\n")
print("Top 5 class-indicative tokens per language:")
for l in LANGS:
    t = top[(top.lang == l) & (top["rank"] <= 5)]
    print(f"  {l}: " + " | ".join(f"{lab[:3]}: {', '.join(t[t.label == lab].token)}" for lab in LABELS))

# ------------------------------------------------------------------ figures
x = np.arange(len(LANGS))

# F1 class balance (train), with test shares as markers
tb = bal[bal.split == "train"].set_index("lang").reindex(LANGS)
ts = bal[bal.split == "test"].set_index("lang").reindex(LANGS)
fig, ax = plt.subplots(figsize=(11, 4.4))
bottom = np.zeros(len(LANGS))
for lab, col in zip(LABELS, [C_NEG, C_NEU, C_POS]):
    ax.bar(x, tb[f"share_{lab}"], 0.6, bottom=bottom, color=col, label=f"{lab} (train)")
    bottom += tb[f"share_{lab}"].values
ax.scatter(x + 0.36, ts.share_negative, marker="_", s=120, color=C_NEG, lw=2, label="test share: negative")
ax.scatter(x + 0.36, ts.share_negative + ts.share_neutral, marker="_", s=120, color="k", lw=2, label="test share: neg + neu")
for i, l in enumerate(LANGS):
    ax.text(i, 101, f"{int(splits.loc[l, 'train']):,}", ha="center", fontsize=7, color="#444")
ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_ylim(0, 108); ax.set_ylabel("% of training tweets")
ax.set_title("Class balance per language (bars: train; ticks: test cumulative shares; numbers: train size)")
ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(.5, -.28)); fig.tight_layout()
fig.savefig(EDA / "eda_class_balance.png", dpi=200); plt.close(fig)

# F2 length distribution
fig, ax = plt.subplots(figsize=(11, 4))
ax.boxplot([df[df.lang == l].n_words.values for l in LANGS], showfliers=False, widths=.55,
           medianprops={"color": "#c55a11"})
ax.set_xticks(x + 1); ax.set_xticklabels(LANGS); ax.set_ylabel("whitespace tokens per tweet")
ax.set_title("Tweet length per language (box = IQR, whiskers = 1.5 IQR, outliers hidden)"); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(EDA / "eda_length.png", dpi=200); plt.close(fig)

# F3 script composition + mixed-script share
fig, ax = plt.subplots(figsize=(11, 4.2))
bottom = np.zeros(len(LANGS))
for c, lab, col in [("pct_latin_letters", "Latin", "#4f81bd"), ("pct_arabic_letters", "Arabic", "#9bbb59"),
                    ("pct_ethiopic_letters", "Ethiopic", "#8064a2")]:
    ax.bar(x, mix[c], .6, bottom=bottom, color=col, label=f"{lab} letters"); bottom += mix[c].values
ax2 = ax.twinx(); ax2.plot(x, mix.pct_tweets_mixed_script, "ko-", ms=4, label="% tweets mixing scripts")
ax2.set_ylabel("% tweets mixing scripts"); ax2.set_ylim(0, max(10, mix.pct_tweets_mixed_script.max() * 1.25))
ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_ylabel("% of letters"); ax.set_ylim(0, 100)
ax.set_title("Writing-system composition per language (@user and URLs removed)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center", bbox_to_anchor=(.5, -.25), ncol=4)
fig.tight_layout(); fig.savefig(EDA / "eda_script_mix.png", dpi=200); plt.close(fig)

# F4 vocabulary: OOV rate vs training size
fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.scatter(voc.train_tokens, voc.test_oov_token_rate, s=40, color="#1f4e79")
for _, r in voc.iterrows():
    ax.annotate(r.lang, (r.train_tokens, r.test_oov_token_rate), xytext=(4, 3), textcoords="offset points", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("training tokens (log scale)"); ax.set_ylabel("test OOV token rate (%)")
ax.set_title("Lexical coverage: share of test tokens unseen in training"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(EDA / "eda_oov.png", dpi=200); plt.close(fig)

# F5 surface features heatmap
cols = ["pct_user", "pct_hashtag", "pct_emoji", "pct_digit", "pct_elongation", "pct_url"]
M = surf[cols].T.values
fig, ax = plt.subplots(figsize=(11, 3.4))
im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(60, np.nanmax(M)))
ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_yticks(range(len(cols)))
ax.set_yticklabels(["@user mention", "hashtag", "emoji", "digit", "elongation (aaa)", "URL"])
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] > 35 else "k")
ax.set_title("Surface features: % of tweets containing each"); plt.colorbar(im, ax=ax, fraction=.02)
fig.tight_layout(); fig.savefig(EDA / "eda_surface.png", dpi=200); plt.close(fig)

if trunc_rows:
    piv = trunc.pivot(index="lang", columns="model", values="subwords_per_word").reindex(LANGS)
    fig, ax = plt.subplots(figsize=(11, 4)); w = .8 / piv.shape[1]
    for k, m in enumerate(piv.columns):
        ax.bar(x + (k - (piv.shape[1] - 1) / 2) * w, piv[m], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(LANGS); ax.set_ylabel("subwords per whitespace token")
    ax.set_title("Tokenizer fertility per language (higher = words split into more pieces)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(EDA / "eda_tokenizer_fertility.png", dpi=200); plt.close(fig)

print(f"\nwritten to {EDA}")
