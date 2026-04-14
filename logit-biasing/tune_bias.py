# coding: utf-8
# tune_bias.py

import sys
sys.path.append('/scratch/user/kiarab/XRec/explainer')
sys.path.append('/scratch/user/kiarab/XRec')
sys.path.append('/scratch/user/kiarab/XRec/logit-biasing')

import pickle
import torch
import numpy as np
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args
from logit_bias import ItemConstrainedLogitsProcessor

# Import evaluation functions directly from eval_biased
from eval_biased import BERT_score, ROUGE_score, BLEU_score, unique_sentence_percent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

# -- Bias combinations to try -------------------------------------------------
BIAS_COMBOS = [
    (0.0,  0.0),   # no biasing (control)
    (0.5,  2.0),   # minimal
    (1.0,  3.0),   # very subtle
    (2.0,  5.0),   # moderate
    (3.0,  5.0),   # current
    (5.0,  5.0),   # equal balanced
    (1.0,  8.0),   # focus on suppressing fillers
    (2.0,  8.0),   # moderate positive, strong negative
]

MAX_SAMPLES = 100


def evaluate(predictions, references):
    bp, br, bf, bp_std, br_std, bf_std = BERT_score(predictions, references)
    rouge_scores                        = ROUGE_score(predictions, references)
    bleu, bleu_precisions               = BLEU_score(predictions, references)
    tokens_predict                      = [s.split() for s in predictions]
    usr, _                              = unique_sentence_percent(tokens_predict)
    avg_len                             = np.mean([len(p.split()) for p in predictions])
    return {
        "bert_precision": bp, "bert_recall": br, "bert_f1": bf,
        "bert_f1_std":    bf_std,
        "rouge1":         rouge_scores['rouge1'],
        "rouge2":         rouge_scores['rouge2'],
        "rougeL":         rouge_scores['rougeL'],
        "bleu4":          bleu,
        "bleu1":          bleu_precisions[0],
        "bleu2":          bleu_precisions[1],
        "bleu3":          bleu_precisions[2],
        "usr":            usr,
        "avg_len":        avg_len,
    }


# -- Load model and data once --------------------------------------------------
print("Loading model and data...")
model = Explainer().to(device)
model.user_embedding_converter.load_state_dict(
    torch.load(f"/scratch/user/kiarab/XRec/data/{args.dataset}/user_converter.pkl", map_location=device)
)
model.item_embedding_converter.load_state_dict(
    torch.load(f"/scratch/user/kiarab/XRec/data/{args.dataset}/item_converter.pkl", map_location=device)
)
model.eval()

data_handler = DataHandler()
_, _, tst_loader = data_handler.load_data()

with open(f"/scratch/user/kiarab/XRec/data/{args.dataset}/item_vocab.pkl", "rb") as f:
    vocab_data    = pickle.load(f)
item_vocab    = vocab_data["item_vocab"]
blacklist_ids = vocab_data["blacklist_ids"]

# Pre-load first MAX_SAMPLES batches
print(f"Pre-loading {MAX_SAMPLES} samples...")
batches = []
for i, batch in enumerate(tst_loader):
    if i >= MAX_SAMPLES:
        break
    batches.append((i, batch))

print(f"Loaded {len(batches)} samples. Starting tuning...\n")


# -- Run experiments -----------------------------------------------------------
all_results = []

for pos_bias, neg_bias in BIAS_COMBOS:
    print(f"Testing positive_bias={pos_bias}, negative_bias={neg_bias}...", flush=True)
    predictions = []
    references  = []

    with torch.no_grad():
        for i, batch in batches:
            user_embed, item_embed, input_text, explain = batch
            user_embed = user_embed.to(device)
            item_embed = item_embed.to(device)

            iid          = str(data_handler.tst_dict["iid"][i])
            verified_ids = item_vocab.get(iid, set())

            if pos_bias == 0.0 and neg_bias == 0.0:
                outputs = model.generate(user_embed, item_embed, input_text)
            else:
                logit_processor = ItemConstrainedLogitsProcessor(
                    verified_token_ids=verified_ids,
                    blacklist_token_ids=blacklist_ids,
                    positive_bias=pos_bias,
                    negative_bias=neg_bias,
                    device=str(device),
                )
                outputs = model.generate(
                    user_embed, item_embed, input_text,
                    logits_processor=[logit_processor]
                )

            end_idx = outputs[0].find("[")
            if end_idx != -1:
                outputs[0] = outputs[0][:end_idx]

            predictions.append(outputs[0])
            references.append(explain[0])

    print(f"  Evaluating...", flush=True)
    scores             = evaluate(predictions, references)
    scores["pos_bias"] = pos_bias
    scores["neg_bias"] = neg_bias
    all_results.append(scores)
    print(f"  ROUGE-1={scores['rouge1']:.4f} BLEU-4={scores['bleu4']:.4f} BERT-F1={scores['bert_f1']:.4f}", flush=True)

# Save results
with open("logit-biasing/tune_results.pkl", "wb") as f:
    pickle.dump(all_results, f)


# -- Print results table -------------------------------------------------------
print("\n")
print("=" * 100)
print("BIAS TUNING RESULTS")
print("=" * 100)
header = (
    f"{'pos':>6} {'neg':>6} | "
    f"{'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} | "
    f"{'BLEU-1':>7} {'BLEU-2':>7} {'BLEU-3':>7} {'BLEU-4':>7} | "
    f"{'BERT-F1':>8} {'STD':>6} | "
    f"{'USR':>6} {'AvgLen':>7}"
)
print(header)
print("-" * 100)

best_bert  = max(all_results, key=lambda x: x["bert_f1"])
best_rouge = max(all_results, key=lambda x: x["rouge1"])
best_bleu  = max(all_results, key=lambda x: x["bleu4"])

for r in all_results:
    markers = []
    if r == best_bert:
        markers.append("best BERT-F1")
    if r == best_rouge:
        markers.append("best ROUGE-1")
    if r == best_bleu:
        markers.append("best BLEU-4")
    marker_str = " <-- " + ", ".join(markers) if markers else ""

    print(
        f"{r['pos_bias']:>6.1f} {r['neg_bias']:>6.1f} | "
        f"{r['rouge1']:>8.4f} {r['rouge2']:>8.4f} {r['rougeL']:>8.4f} | "
        f"{r['bleu1']:>7.4f} {r['bleu2']:>7.4f} {r['bleu3']:>7.4f} {r['bleu4']:>7.4f} | "
        f"{r['bert_f1']:>8.4f} {r['bert_f1_std']:>6.4f} | "
        f"{r['usr']:>6.4f} {r['avg_len']:>7.1f}"
        f"{marker_str}"
    )

print("=" * 100)
print(f"\nResults saved to logit-biasing/tune_results.pkl")