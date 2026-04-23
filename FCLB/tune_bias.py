import sys
sys.path.append('/scratch/user/kiarab/XRec/explainer')
sys.path.append('/scratch/user/kiarab/XRec')
sys.path.append('/scratch/user/kiarab/XRec/logit-biasing')

import pickle
import json
import os
import torch
import numpy as np
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args
from logit_bias import ItemConstrainedLogitsProcessor

from eval_biased import (
    BERT_score, ROUGE_score, BLEU_score,
    unique_sentence_percent, preference_consistency_score,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

BIAS_COMBOS = [
    # (positive_bias, negative_bias, min_position)
    (0.0,   0.0,  8),    # control — no biasing
    (0.05,  0.0,  8),    # very subtle positive only
    (0.10,  0.0,  8),    # subtle positive only
    (0.15,  0.0,  8),    # mild positive only
    (0.20,  0.0,  8),    # moderate positive only
    (0.30,  0.0,  8),    # stronger positive only
    (0.50,  0.0,  8),    # upper bound positive only
    (0.15,  0.0,  4),    # mild positive, earlier activation
    (0.15,  0.0, 12),    # mild positive, later activation
    (0.10,  0.1,  8),    # subtle positive + very mild suppression
    (0.15,  0.2,  8),    # mild positive + mild suppression
    (0.20,  0.3,  8),    # moderate positive + moderate suppression
]

MAX_SAMPLES = 100


def evaluate_full(predictions, references, item_words_list=None,
                  user_words_list=None):
    """Run all metrics including preference consistency."""
    bp, br, bf, bp_std, br_std, bf_std = BERT_score(predictions, references)
    rouge_scores = ROUGE_score(predictions, references)
    bleu, bleu_precisions = BLEU_score(predictions, references)
    tokens_predict = [s.split() for s in predictions]
    usr, _ = unique_sentence_percent(tokens_predict)
    avg_len = np.mean([len(p.split()) for p in predictions])

    result = {
        "bert_precision": bp, "bert_recall": br, "bert_f1": bf,
        "bert_f1_std": bf_std,
        "rouge1": rouge_scores['rouge1'],
        "rouge2": rouge_scores['rouge2'],
        "rougeL": rouge_scores['rougeL'],
        "bleu4": bleu,
        "bleu1": bleu_precisions[0],
        "bleu2": bleu_precisions[1],
        "bleu3": bleu_precisions[2],
        "usr": usr,
        "avg_len": avg_len,
    }

    # Preference consistency — the metric that answers the research question
    if item_words_list is not None:
        pref_mean, pref_per_sample = preference_consistency_score(
            predictions, item_words_list, user_words_list,
        )
        result["pref_consistency"] = pref_mean
        result["pref_consistency_std"] = float(np.std(pref_per_sample))
    else:
        result["pref_consistency"] = None
        result["pref_consistency_std"] = None

    return result


# -- Load model and data once --------------------------------------------------
print("Loading model and data...")
model = Explainer().to(device)
model.user_embedding_converter.load_state_dict(
    torch.load(
        f"/scratch/user/kiarab/XRec/data/{args.dataset}/user_converter.pkl",
        map_location=device,
    )
)
model.item_embedding_converter.load_state_dict(
    torch.load(
        f"/scratch/user/kiarab/XRec/data/{args.dataset}/item_converter.pkl",
        map_location=device,
    )
)
model.eval()

data_handler = DataHandler()
_, _, tst_loader = data_handler.load_data()

base = f"/scratch/user/kiarab/XRec/data/{args.dataset}"

with open(f"{base}/item_vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)
item_vocab = vocab_data["item_vocab"]
blacklist_ids = vocab_data["blacklist_ids"]
item_words_dict = vocab_data.get("item_words", {})
user_words_dict = vocab_data.get("user_words", {})

# Load test records for iid/uid mapping
test_records = []
test_data_path = f"{base}/tst.json"
if os.path.exists(test_data_path):
    with open(test_data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))

# Pre-load first MAX_SAMPLES batches
print(f"Pre-loading {MAX_SAMPLES} samples...")
batches = []
for i, batch in enumerate(tst_loader):
    if i >= MAX_SAMPLES:
        break
    batches.append((i, batch))

print(f"Loaded {len(batches)} samples. Starting tuning...\n")


all_results = []

for combo in BIAS_COMBOS:
    pos_bias, neg_bias, min_pos = combo
    print(
        f"Testing pos={pos_bias}, neg={neg_bias}, min_pos={min_pos}...",
        flush=True,
    )
    predictions = []
    references = []
    sample_item_words = []
    sample_user_words = []

    with torch.no_grad():
        for i, batch in batches:
            user_embed, item_embed, input_text, explain