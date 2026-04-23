import sys
sys.path.append('/scratch/user/kiarab/XRec/evaluation')

import numpy as np
import pickle
import argparse
import json
import os
import nltk

nltk.download('punkt', quiet=True)


def BERT_score(predictions, references):
    from bert_score import score
    P, R, F = score(
        predictions, references, lang="en",
        rescale_with_baseline=True, verbose=True,
    )
    return (
        P.mean().item(), R.mean().item(), F.mean().item(),
        P.std().item(),  R.std().item(),  F.std().item(),
    )


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
    refs  = [[r.split()] for r in references]
    preds = [p.split() for p in predictions]
    smoothie = SmoothingFunction().method1
    bleu1 = corpus_bleu(refs, preds, weights=(1, 0, 0, 0),
                        smoothing_function=smoothie)
    bleu2 = corpus_bleu(refs, preds, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoothie)
    bleu3 = corpus_bleu(refs, preds, weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=smoothie)
    bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothie)
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
    Measures what fraction of known preference words appear in
    each generated explanation.

    For each sample i:
      - Let P_i = set of preference words (user ∩ item if user_words
        available, else item_words alone)
      - Let G_i = set of words in the generated explanation
      - consistency_i = |P_i ∩ G_i| / |P_i|  (recall of pref words)

    Returns:
      mean_consistency:  average across all samples
      per_sample:        list of per-sample scores
    """
    per_sample = []

    for i, pred in enumerate(predictions):
        # Determine the preference word set for this sample
        i_words = item_words_list[i] if i < len(item_words_list) else set()

        if user_words_list and i < len(user_words_list):
            u_words = user_words_list[i]
            # Intersection = preference-aligned words
            pref_words = i_words & u_words if u_words else i_words
        else:
            pref_words = i_words

        if not pref_words:
            per_sample.append(1.0)  # no constraint → trivially consistent
            continue

        # Tokenize the generated prediction
        gen_words = set(w.lower() for w in nltk.word_tokenize(pred))

        # Recall: what fraction of preference words appear in output?
        hits = len(pref_words & gen_words)
        consistency = hits / len(pref_words)
        per_sample.append(consistency)

    mean_consistency = np.mean(per_sample) if per_sample else 0.0
    return mean_consistency, per_sample



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--pred", type=str, default=None)
    parser.add_argument("--ref",  type=str, default=None)
    parser.add_argument("--vocab", type=str, default=None,
                        help="Path to item_vocab.pkl for preference consistency")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test JSONL with iid/uid per sample")
    args = parser.parse_args()

    base = f"/scratch/user/kiarab/XRec/data/{args.dataset}"     # Need to add as arg

    pred_path  = args.pred  or f"{base}/tst_predictions_biased.pkl"
    ref_path   = args.ref   or f"{base}/tst_references_biased.pkl"
    vocab_path = args.vocab or f"{base}/item_vocab.pkl"

    with open(pred_path, "rb") as f:
        predictions = pickle.load(f)
    with open(ref_path, "rb") as f:
        references = pickle.load(f)

    print(f"Loaded {len(predictions)} predictions")
    print(f"Predictions: {pred_path}")
    print(f"References:  {ref_path}")

    # -- Standard metrics --
    print("Running BERTScore...")
    bp, br, bf, bp_std, br_std, bf_std = BERT_score(predictions, references)

    print("Running ROUGE...")
    rouge_scores = ROUGE_score(predictions, references)

    print("Running BLEU...")
    bleu, bleu_precisions = BLEU_score(predictions, references)

    tokens_predict = [s.split() for s in predictions]
    usr, _ = unique_sentence_percent(tokens_predict)
    avg_len = np.mean([len(p.split()) for p in predictions])

    # -- Preference consistency (new) --
    pref_consistency = None
    if os.path.exists(vocab_path):
        print("Running Preference Consistency...")
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)

        item_words_dict = vocab_data.get("item_words", {})
        user_words_dict = vocab_data.get("user_words", {})

        # Build per-sample word lists aligned with predictions
        # If test_data JSONL is provided, use it to map index → iid/uid
        item_words_list = []
        user_words_list = []

        test_data_path = args.test_data or f"{base}/tst.json"
        if os.path.exists(test_data_path):
            test_records = []
            with open(test_data_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        test_records.append(json.loads(line))

            for idx in range(len(predictions)):
                if idx < len(test_records):
                    rec = test_records[idx]
                    iid = str(rec.get("iid", ""))
                    uid = str(rec.get("uid", rec.get("user_id", "")))
                    item_words_list.append(item_words_dict.get(iid, set()))
                    user_words_list.append(user_words_dict.get(uid, set()))
                else:
                    item_words_list.append(set())
                    user_words_list.append(set())

            pref_consistency, per_sample_pref = preference_consistency_score(
                predictions, item_words_list,
                user_words_list if user_words_dict else None,
            )
        else:
            print(f"  WARNING: {test_data_path} not found. "
                  f"Skipping preference consistency.")
    else:
        print(f"  WARNING: {vocab_path} not found. "
              f"Skipping preference consistency.")

    # -- Print results --
    print(f"\nDataset: {args.dataset}")
    print("=" * 50)
    print("Evaluation Metrics:")

    print(f"\nBERTScore:")
    print(f"  precision:  {bp:.4f} (+/- {bp_std:.4f})")
    print(f"  recall:     {br:.4f} (+/- {br_std:.4f})")
    print(f"  f1:         {bf:.4f} (+/- {bf_std:.4f})")

    print(f"\nROUGE:")
    print(f"  rouge1:     {rouge_scores['rouge1']:.4f}")
    print(f"  rouge2:     {rouge_scores['rouge2']:.4f}")
    print(f"  rougeL:     {rouge_scores['rougeL']:.4f}")

    print(f"\nBLEU:")
    print(f"  bleu:       {bleu:.4f}")
    print(f"  1-gram:     {bleu_precisions[0]:.4f}")
    print(f"  2-gram:     {bleu_precisions[1]:.4f}")
    print(f"  3-gram:     {bleu_precisions[2]:.4f}")
    print(f"  4-gram:     {bleu_precisions[3]:.4f}")

    print(f"\nDiversity:")
    print(f"  USR:        {usr:.4f}")
    print(f"  avg length: {avg_len:.1f} words")

    if pref_consistency is not None:
        print(f"\nPreference Consistency (NEW):")
        print(f"  mean:       {pref_consistency:.4f}")
        print(f"  std:        {np.std(per_sample_pref):.4f}")
        print(f"  min:        {np.min(per_sample_pref):.4f}")
        print(f"  max:        {np.max(per_sample_pref):.4f}")