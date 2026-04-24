import sys
sys.path.append('/scratch/user/kiarab/XRec/evaluation')

import numpy as np
import pickle
import argparse
import json
import os
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

nltk.download('punkt', quiet=True)


# ----------------------------
# TEXT HELPERS
# ----------------------------
def tokenize(text):
    return nltk.word_tokenize(text.lower())


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        if seq not in unique_seq:
            unique_seq.append(seq)
    return len(unique_seq) / len(sequence_batch), len(unique_seq)


# ----------------------------
# BERT SCORE
# ----------------------------
def BERT_score(predictions, references):
    from bert_score import score

    # safety cleanup
    references = [r if r.strip() != "" else "empty" for r in references]

    P, R, F = score(
        predictions,
        references,
        lang="en",
        rescale_with_baseline=True,
        verbose=True,
    )
    return (
        P.mean().item(), R.mean().item(), F.mean().item(),
        P.std().item(), R.std().item(), F.std().item(),
    )


# ----------------------------
# ROUGE
# ----------------------------
def ROUGE_score(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1, r2, rl = [], [], []
    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rl.append(s['rougeL'].fmeasure)

    return {
        "rouge1": np.mean(r1),
        "rouge2": np.mean(r2),
        "rougeL": np.mean(rl),
    }


# ----------------------------
# BLEU
# ----------------------------
def BLEU_score(predictions, references):
    refs = [[r.split()] for r in references]
    preds = [p.split() for p in predictions]

    smoothie = SmoothingFunction().method1

    bleu1 = corpus_bleu(refs, preds, weights=(1,0,0,0), smoothing_function=smoothie)
    bleu2 = corpus_bleu(refs, preds, weights=(0.5,0.5,0,0), smoothing_function=smoothie)
    bleu3 = corpus_bleu(refs, preds, weights=(0.33,0.33,0.33,0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(refs, preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)

    return bleu4, [bleu1, bleu2, bleu3, bleu4]


# ----------------------------
# FACTUAL PRECISION
# ----------------------------
def factual_precision(predictions, item_words_list):
    scores = []

    for i, pred in enumerate(predictions):
        pred_words = set(tokenize(pred))
        item_words = item_words_list[i] if i < len(item_words_list) else set()

        if not item_words:
            scores.append(1.0)
            continue

        overlap = len(pred_words & item_words) / max(len(pred_words), 1)
        scores.append(overlap)

    return float(np.mean(scores)), float(np.std(scores)), scores


# ----------------------------
# PREF CONSISTENCY
# ----------------------------
def preference_consistency(predictions, item_words_list, user_words_list):
    scores = []

    for i, pred in enumerate(predictions):
        gen_words = set(tokenize(pred))

        item_words = item_words_list[i] if i < len(item_words_list) else set()
        user_words = user_words_list[i] if i < len(user_words_list) else set()

        pref_words = item_words & user_words if user_words else item_words

        if not pref_words:
            scores.append(1.0)
            continue

        scores.append(len(pref_words & gen_words) / len(pref_words))

    return float(np.mean(scores)), float(np.std(scores)), scores


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--pred", type=str, default=None)
    parser.add_argument("--ref", type=str, default=None)
    args = parser.parse_args()

    base = f"/scratch/user/kiarab/XRec/FCLB/data/{args.dataset}"

    pred_path = args.pred or f"{base}/tst_predictions_biased_speed.pkl"
    ref_path  = args.ref  or f"{base}/tst_references_biased_speed.pkl"
    vocab_path = f"{base}/item_vocab.pkl"
    test_path  = f"{base}/tst.pkl"

    # ----------------------------
    # LOAD PRED/REF
    # ----------------------------
    with open(pred_path, "rb") as f:
        predictions = pickle.load(f)

    with open(ref_path, "rb") as f:
        references = pickle.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # ----------------------------
    # LOAD VOCAB
    # ----------------------------
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    item_words_dict = vocab.get("item_words", {})
    user_words_dict = vocab.get("user_words", {})

    # ----------------------------
    # LOAD TEST DATA (SAFE)
    # ----------------------------
    item_words_list = []
    user_words_list = []

    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            test_data = pickle.load(f)

        # pandas DataFrame case (XRec standard)
        if hasattr(test_data, "to_dict"):
            test_dict = test_data.to_dict("list")
        else:
            raise ValueError("tst.pkl is not a DataFrame. Check format.")

        for i in range(len(predictions)):
            iid = str(test_dict["iid"][i])
            uid = str(test_dict["uid"][i])

            item_words_list.append(item_words_dict.get(iid, set()))
            user_words_list.append(user_words_dict.get(uid, set()))

    else:
        raise FileNotFoundError(f"{test_path} not found (needed for alignment)")

    # ----------------------------
    # METRICS
    # ----------------------------
    print("\nRunning BERTScore...")
    bp, br, bf, bp_std, br_std, bf_std = BERT_score(predictions, references)

    print("Running ROUGE...")
    rouge = ROUGE_score(predictions, references)

    print("Running BLEU...")
    bleu, bleu_parts = BLEU_score(predictions, references)

    print("Running diversity...")
    usr, _ = unique_sentence_percent([p.split() for p in predictions])

    print("Running factual precision...")
    fp, fp_std, _ = factual_precision(predictions, item_words_list)

    print("Running preference consistency...")
    pc, pc_std, _ = preference_consistency(predictions, item_words_list, user_words_list)

    avg_len = np.mean([len(p.split()) for p in predictions])

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(f"BERT F1: {bf:.4f} ± {bf_std:.4f}")

    print("\nROUGE:")
    print(rouge)

    print(f"\nBLEU-4: {bleu:.4f}")

    print("\nDiversity:")
    print(f"USR: {usr:.4f}")
    print(f"Avg length: {avg_len:.1f}")

    print("\nFactual Precision:")
    print(f"{fp:.4f} ± {fp_std:.4f}")

    print("\nPreference Consistency:")
    print(f"{pc:.4f} ± {pc_std:.4f}")

    # ----------------------------
    # SAVE RESULTS
    # ----------------------------
    out_path = f"{base}/eval_results.json"

    results = {
        "bert_f1": bf,
        "rouge": rouge,
        "bleu": bleu,
        "usr": usr,
        "factual_precision": fp,
        "preference_consistency": pc
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")