import sys
sys.path.append('/scratch/user/kiarab/XRec/explainer')
sys.path.append('/scratch/user/kiarab/XRec')
sys.path.append('/scratch/user/kiarab/XRec/FCLB')

import pickle
import json
import os
import torch
import numpy as np
import nltk

nltk.download('punkt', quiet=True)

from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args
from logit_bias import ItemConstrainedLogitsProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")


# -- Bias combinations ---------------------------------------------------------
BIAS_COMBOS = [
    # (positive_bias, negative_bias, min_position)
    (0.0,   0.0,  8),    # control -- no biasing
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


# -- Evaluation functions (inline to avoid import issues) ----------------------

def ROUGE_score(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True,
    )
    rouge1, rouge2, rougeL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        rouge1.append(s['rouge1'].fmeasure)
        rouge2.append(s['rouge2'].fmeasure)
        rougeL.append(s['rougeL'].fmeasure)
    return {
        'rouge1': np.mean(rouge1),
        'rouge2': np.mean(rouge2),
        'rougeL': np.mean(rougeL),
    }


def BLEU_score(predictions, references):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs = [[r.split()] for r in references]
    preds = [p.split() for p in predictions]
    sm = SmoothingFunction().method1
    bleu1 = corpus_bleu(refs, preds, weights=(1, 0, 0, 0), smoothing_function=sm)
    bleu2 = corpus_bleu(refs, preds, weights=(0.5, 0.5, 0, 0), smoothing_function=sm)
    bleu3 = corpus_bleu(refs, preds, weights=(0.33, 0.33, 0.33, 0), smoothing_function=sm)
    bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sm)
    return bleu4, [bleu1, bleu2, bleu3, bleu4]


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)
    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def preference_consistency_score(predictions, item_words_list,
                                  user_words_list=None):
    """
    Measures what fraction of known preference words appear in each
    generated explanation.
    consistency_i = |P_i intersection G_i| / |P_i|
    """
    per_sample = []
    for i, pred in enumerate(predictions):
        i_words = item_words_list[i] if i < len(item_words_list) else set()
        if user_words_list and i < len(user_words_list):
            u_words = user_words_list[i]
            pref_words = i_words & u_words if u_words else i_words
        else:
            pref_words = i_words
        if not pref_words:
            per_sample.append(1.0)
            continue
        gen_words = set(w.lower() for w in nltk.word_tokenize(pred))
        hits = len(pref_words & gen_words)
        consistency = hits / len(pref_words)
        per_sample.append(consistency)
    mean_consistency = np.mean(per_sample) if per_sample else 0.0
    return mean_consistency, per_sample


def evaluate_fast(predictions, references, item_words_list=None,
                  user_words_list=None):
    """Speed-optimized: skips BERTScore, uses ROUGE + BLEU + PrefConsistency."""
    rouge_scores = ROUGE_score(predictions, references)
    bleu, bleu_precisions = BLEU_score(predictions, references)
    tokens_predict = [s.split() for s in predictions]
    usr, _ = unique_sentence_percent(tokens_predict)
    avg_len = np.mean([len(p.split()) for p in predictions])

    result = {
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

    if item_words_list is not None and len(item_words_list) > 0:
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

base = f"/scratch/user/kiarab/XRec/FCLB/data/{args.dataset}"

model.user_embedding_converter.load_state_dict(
    torch.load(f"{base}/user_converter.pkl", map_location=device)
)
model.item_embedding_converter.load_state_dict(
    torch.load(f"{base}/item_converter.pkl", map_location=device)
)
model.eval()

data_handler = DataHandler()
_, _, tst_loader = data_handler.load_data()

with open(f"{base}/item_vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)

item_vocab = vocab_data["item_vocab"]
blacklist_ids = vocab_data["blacklist_ids"]
item_words_dict = vocab_data.get("item_words", {})
user_words_dict = vocab_data.get("user_words", {})

has_word_dicts = len(item_words_dict) > 0
if has_word_dicts:
    print(f"Found item_words for {len(item_words_dict)} items")
    print(f"Found user_words for {len(user_words_dict)} users")
else:
    print("WARNING: item_words/user_words not in vocab pickle.")
    print("  Preference consistency will be skipped.")

# Load test records for iid/uid mapping
test_records = []
test_data_path = f"{base}/tst.json"
if os.path.exists(test_data_path):
    with open(test_data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))
    print(f"Loaded {len(test_records)} test records from {test_data_path}")
else:
    print(f"WARNING: {test_data_path} not found. Using index-based iid lookup.")

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
            user_embed, item_embed, input_text, explain = batch
            user_embed = user_embed.to(device)
            item_embed = item_embed.to(device)

            # Get item/user IDs
            if i < len(test_records):
                iid = str(test_records[i].get("iid", ""))
                uid = str(test_records[i].get("uid",
                          test_records[i].get("user_id", "")))
            else:
                iid = str(data_handler.tst_dict["iid"][i])
                uid = ""

            verified_ids = item_vocab.get(iid, set())

            # Build per-sample word lists for preference consistency
            if has_word_dicts:
                sample_item_words.append(item_words_dict.get(iid, set()))
                sample_user_words.append(user_words_dict.get(uid, set()))

            # Generate
            if pos_bias == 0.0 and neg_bias == 0.0:
                outputs = model.generate(user_embed, item_embed, input_text)
            else:
                logit_processor = ItemConstrainedLogitsProcessor(
                    verified_token_ids=verified_ids,
                    blacklist_token_ids=blacklist_ids,
                    positive_bias=pos_bias,
                    negative_bias=neg_bias,
                    min_position=min_pos,
                    device=str(device),
                )
                outputs = model.generate(
                    user_embed, item_embed, input_text,
                    logits_processor=[logit_processor],
                )

            # Clean up output
            end_idx = outputs[0].find("[")
            if end_idx != -1:
                outputs[0] = outputs[0][:end_idx]

            predictions.append(outputs[0])
            references.append(explain[0])

            if i % 25 == 0:
                print(f"    sample {i}/{len(batches)}...", flush=True)

    # Evaluate this combination
    print(f"  Evaluating...", flush=True)
    scores = evaluate_fast(
        predictions, references,
        item_words_list=sample_item_words if has_word_dicts else None,
        user_words_list=sample_user_words if has_word_dicts else None,
    )
    scores["pos_bias"] = pos_bias
    scores["neg_bias"] = neg_bias
    scores["min_pos"] = min_pos
    all_results.append(scores)

    pref_str = ""
    if scores["pref_consistency"] is not None:
        pref_str = f" PREF={scores['pref_consistency']:.4f}"

    print(
        f"  ROUGE-1={scores['rouge1']:.4f}"
        f" BLEU-4={scores['bleu4']:.4f}"
        f"{pref_str}",
        flush=True,
    )
    print()


# -- Save results --------------------------------------------------------------
os.makedirs("FCLB", exist_ok=True)
results_path = "FCLB/tune_results.pkl"

with open(results_path, "wb") as f:
    pickle.dump(all_results, f)


# -- Print results table -------------------------------------------------------
print("\n")
print("=" * 120)
print("BIAS TUNING RESULTS (BERTScore skipped for speed -- run on winner only)")
print("=" * 120)

has_pref = any(r["pref_consistency"] is not None for r in all_results)

header = (
    f"{'pos':>6} {'neg':>6} {'minP':>5} | "
    f"{'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} | "
    f"{'BLEU-1':>7} {'BLEU-4':>7} | "
    f"{'USR':>6} {'AvgLen':>7}"
)
if has_pref:
    header += f" | {'PrefCon':>8} {'Std':>6}"
print(header)
print("-" * 120)

best_rouge = max(all_results, key=lambda x: x["rouge1"])
best_bleu = max(all_results, key=lambda x: x["bleu4"])
best_pref = None
if has_pref:
    pref_results = [r for r in all_results if r["pref_consistency"] is not None]
    if pref_results:
        best_pref = max(pref_results, key=lambda x: x["pref_consistency"])

for r in all_results:
    markers = []
    if r is best_rouge:
        markers.append("best ROUGE-1")
    if r is best_bleu:
        markers.append("best BLEU-4")
    if best_pref is not None and r is best_pref:
        markers.append("best PrefCon")
    marker_str = " <-- " + ", ".join(markers) if markers else ""

    line = (
        f"{r['pos_bias']:>6.2f} {r['neg_bias']:>6.2f} {r['min_pos']:>5d} | "
        f"{r['rouge1']:>8.4f} {r['rouge2']:>8.4f} {r['rougeL']:>8.4f} | "
        f"{r['bleu1']:>7.4f} {r['bleu4']:>7.4f} | "
        f"{r['usr']:>6.4f} {r['avg_len']:>7.1f}"
    )
    if has_pref:
        pc = r["pref_consistency"]
        ps = r["pref_consistency_std"]
        if pc is not None:
            line += f" | {pc:>8.4f} {ps:>6.4f}"
        else:
            line += f" | {'N/A':>8} {'N/A':>6}"
    line += marker_str
    print(line)

print("=" * 120)
print(f"\nResults saved to {results_path}")