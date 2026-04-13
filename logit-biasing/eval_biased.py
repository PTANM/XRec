# coding: utf-8
# eval_biased.py

import sys
sys.path.append('/scratch/user/kiarab/XRec/evaluation')

import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="amazon")
parser.add_argument("--pred", type=str, default=None, help="Path to predictions pkl file")
parser.add_argument("--ref",  type=str, default=None, help="Path to references pkl file")
args = parser.parse_args()

# Default to biased predictions if no paths provided
pred_path = args.pred if args.pred else f"/scratch/user/kiarab/XRec/data/{args.dataset}/tst_predictions_biased.pkl"
ref_path  = args.ref  if args.ref  else f"/scratch/user/kiarab/XRec/data/{args.dataset}/tst_references_biased.pkl"

with open(pred_path, "rb") as f:
    predictions = pickle.load(f)
with open(ref_path, "rb") as f:
    references = pickle.load(f)

print(f"Loaded {len(predictions)} predictions")
print(f"Predictions: {pred_path}")
print(f"References:  {ref_path}")


# -- BERTScore ----------------------------------------------------------------
def BERT_score(predictions, references):
    from bert_score import score
    P, R, F = score(predictions, references, lang="en", rescale_with_baseline=True, verbose=True)
    return (
        P.mean().item(), R.mean().item(), F.mean().item(),
        P.std().item(),  R.std().item(),  F.std().item(),
    )


# -- ROUGE --------------------------------------------------------------------
def ROUGE_score(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
    return {
        'rouge1': np.mean(rouge1),
        'rouge2': np.mean(rouge2),
        'rougeL': np.mean(rougeL),
    }


# -- BLEU ---------------------------------------------------------------------
def BLEU_score(predictions, references):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    refs  = [[r.split()] for r in references]
    preds = [p.split() for p in predictions]
    smoothie = SmoothingFunction().method1
    bleu1 = corpus_bleu(refs, preds, weights=(1,0,0,0),            smoothing_function=smoothie)
    bleu2 = corpus_bleu(refs, preds, weights=(0.5,0.5,0,0),        smoothing_function=smoothie)
    bleu3 = corpus_bleu(refs, preds, weights=(0.33,0.33,0.33,0),   smoothing_function=smoothie)
    bleu4 = corpus_bleu(refs, preds, weights=(0.25,0.25,0.25,0.25),smoothing_function=smoothie)
    return bleu4, [bleu1, bleu2, bleu3, bleu4]


# -- USR ----------------------------------------------------------------------
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


# -- Run Evaluation -----------------------------------------------------------
print("Running BERTScore...")
bp, br, bf, bp_std, br_std, bf_std = BERT_score(predictions, references)

print("Running ROUGE...")
rouge_scores = ROUGE_score(predictions, references)

print("Running BLEU...")
bleu, bleu_precisions = BLEU_score(predictions, references)

tokens_predict = [s.split() for s in predictions]
usr, _ = unique_sentence_percent(tokens_predict)
avg_len = np.mean([len(p.split()) for p in predictions])

print(f"\nDataset: {args.dataset}")
print("=" * 40)
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