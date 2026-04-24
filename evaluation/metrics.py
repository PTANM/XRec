import argparse
import concurrent.futures
import json
import math
import os
import re
from collections import Counter
from typing import Iterable, List

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BartForConditionalGeneration


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "book",
    "books",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "read",
    "reader",
    "readers",
    "story",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "user",
    "was",
    "were",
    "will",
    "with",
    "would",
}

GENERIC_UNIGRAMS = {
    "author",
    "book",
    "books",
    "character",
    "characters",
    "experience",
    "novel",
    "novels",
    "reader",
    "readers",
    "review",
    "series",
    "story",
    "writing",
}


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="amazon", help="amazon, yelp or google")
parser.add_argument(
    "--results_tag",
    type=str,
    default="mlp",
    help="Run tag used by explainer/main.py when it wrote the structured results.",
)
parser.add_argument(
    "--results_path",
    type=str,
    default="",
    help="Optional explicit path to a structured JSONL results file.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size for BARTScore and LlamaScore.",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional limit on the number of evaluation examples to load.",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=512,
    help="Maximum token length for semantic scoring models.",
)
parser.add_argument(
    "--bart_model_name",
    type=str,
    default="facebook/bart-large-cnn",
    help="Sequence-to-sequence model used for BARTScore.",
)
parser.add_argument(
    "--llama_model_name",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Causal LM used for the local LlamaScore.",
)
parser.add_argument(
    "--llama_load_in_8bit",
    action="store_true",
    help="Load the LlamaScore model in 8-bit mode when CUDA is available.",
)
parser.add_argument(
    "--skip_bart_score",
    action="store_true",
    help="Skip BARTScore evaluation.",
)
parser.add_argument(
    "--skip_llama_score",
    action="store_true",
    help="Skip local LlamaScore evaluation.",
)
parser.add_argument(
    "--metadata_path",
    type=str,
    default="",
    help="Optional JSON or JSONL file containing raw item metadata keyed by iid.",
)
parser.add_argument(
    "--gpt_judge_model",
    type=str,
    default="",
    help="Optional OpenAI-compatible judge model name, e.g. gpt-5.2.",
)
parser.add_argument(
    "--gpt_judge_base_url",
    type=str,
    default="",
    help="Optional OpenAI-compatible base URL, useful for TAMU Chat or proxies.",
)
parser.add_argument(
    "--gpt_judge_api_key_env",
    type=str,
    default="OPENAI_API_KEY",
    help="Environment variable containing the GPT judge API key.",
)
parser.add_argument(
    "--gpt_judge_max_workers",
    type=int,
    default=8,
    help="Maximum parallel judge requests for GPT-based scoring.",
)
parser.add_argument(
    "--gpt_judge_system_prompt_path",
    type=str,
    default="evaluation/system_prompt.txt",
    help="System prompt used by the GPT-style judge.",
)
args = parser.parse_args()


def huggingface_load_kwargs():
    kwargs = {}
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token

    if any(
        os.environ.get(flag) == "1"
        for flag in ("HF_LOCAL_FILES_ONLY", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        kwargs["local_files_only"] = True

    return kwargs


def batch_items(items: List, batch_size: int) -> Iterable[List]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def score_mean_std(values):
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array))


def normalize_text(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [token for token in cleaned.split() if token]


def ngrams(tokens, size):
    if len(tokens) < size:
        return []
    return [tuple(tokens[idx : idx + size]) for idx in range(len(tokens) - size + 1)]


def extract_attribute_ngrams(text: str, max_ngram: int = 4):
    tokens = [token for token in normalize_text(text) if token not in STOPWORDS]
    attributes = set()
    for ngram_size in range(1, max_ngram + 1):
        for idx in range(len(tokens) - ngram_size + 1):
            ngram = tokens[idx : idx + ngram_size]
            if ngram_size == 1 and (
                len(ngram[0]) < 5 or ngram[0] in GENERIC_UNIGRAMS
            ):
                continue
            attributes.add(" ".join(ngram))
    return attributes


def load_metadata_override(metadata_path):
    if not metadata_path:
        return {}
    with open(metadata_path, "r") as file:
        content = file.read().strip()
    if not content:
        return {}

    if content[0] == "{":
        parsed = json.loads(content)
        if all(isinstance(key, str) for key in parsed.keys()):
            return {int(key): str(value) for key, value in parsed.items()}

    metadata = {}
    for line in content.splitlines():
        row = json.loads(line)
        iid = int(row["iid"])
        if "item_metadata_text" in row:
            metadata[iid] = row["item_metadata_text"]
        elif "metadata" in row:
            metadata[iid] = row["metadata"]
        elif "content" in row:
            value = row["content"]
            metadata[iid] = value if isinstance(value, str) else json.dumps(value)
        else:
            metadata[iid] = json.dumps(row)
    return metadata


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


def bert_score(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
    )
    precision = np.asarray(results["precision"])
    recall = np.asarray(results["recall"])
    f1 = np.asarray(results["f1"])
    return (
        float(np.mean(precision)),
        float(np.mean(recall)),
        float(np.mean(f1)),
        float(np.std(precision)),
        float(np.std(recall)),
        float(np.std(f1)),
    )


def rouge_n_scores(predictions, references, ngram_size):
    scores = []
    for prediction, reference in zip(predictions, references):
        prediction_counts = Counter(ngrams(normalize_text(prediction), ngram_size))
        reference_counts = Counter(ngrams(normalize_text(reference), ngram_size))

        if not prediction_counts or not reference_counts:
            scores.append(0.0)
            continue

        overlap = sum((prediction_counts & reference_counts).values())
        precision = overlap / sum(prediction_counts.values())
        recall = overlap / sum(reference_counts.values())
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return score_mean_std(scores)


def lcs_length(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0

    previous = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current = [0]
        for idx, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_scores(predictions, references):
    scores = []
    for prediction, reference in zip(predictions, references):
        prediction_tokens = normalize_text(prediction)
        reference_tokens = normalize_text(reference)
        if not prediction_tokens or not reference_tokens:
            scores.append(0.0)
            continue

        overlap = lcs_length(prediction_tokens, reference_tokens)
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return score_mean_std(scores)


def corpus_bleu_scores(predictions, references, max_order=4):
    prediction_lengths = 0
    reference_lengths = 0
    precisions = []

    for ngram_size in range(1, max_order + 1):
        clipped_matches = 0
        total_candidates = 0
        for prediction, reference in zip(predictions, references):
            prediction_tokens = normalize_text(prediction)
            reference_tokens = normalize_text(reference)
            if ngram_size == 1:
                prediction_lengths += len(prediction_tokens)
                reference_lengths += len(reference_tokens)

            prediction_counts = Counter(ngrams(prediction_tokens, ngram_size))
            reference_counts = Counter(ngrams(reference_tokens, ngram_size))
            clipped_matches += sum(
                min(count, reference_counts[ngram])
                for ngram, count in prediction_counts.items()
            )
            total_candidates += sum(prediction_counts.values())

        if total_candidates == 0:
            precisions.append(0.0)
        elif clipped_matches == 0:
            precisions.append(1.0 / (total_candidates + 1.0))
        else:
            precisions.append(clipped_matches / total_candidates)

    if prediction_lengths == 0:
        brevity_penalty = 0.0
    elif prediction_lengths > reference_lengths:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (reference_lengths / prediction_lengths))

    bleu_scores = []
    for order in range(1, max_order + 1):
        order_precisions = precisions[:order]
        if any(value <= 0 for value in order_precisions):
            bleu_scores.append(0.0)
            continue
        bleu_scores.append(
            brevity_penalty
            * math.exp(sum(math.log(value) for value in order_precisions) / order)
        )

    return {
        "bleu": float(bleu_scores[-1]),
        "bleu1": float(bleu_scores[0]),
        "bleu2": float(bleu_scores[1]),
        "bleu3": float(bleu_scores[2]),
        "bleu4": float(bleu_scores[3]),
    }


def average_prediction_length(predictions):
    lengths = [len(normalize_text(prediction)) for prediction in predictions]
    return float(np.mean(np.asarray(lengths))) if lengths else 0.0


def extract_numeric_score(text):
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        raise ValueError(f"Could not parse numeric score from judge response: {text!r}")
    value = float(match.group(0))
    return max(0.0, min(100.0, value))


class GptJudgeScorer:
    def __init__(self, model_name, api_key, base_url="", max_workers=8):
        if not api_key:
            raise ValueError("GPT judge API key is required when gpt_judge_model is set.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.max_workers = max_workers

        with open(args.gpt_judge_system_prompt_path, "r") as file:
            self.system_prompt = file.read().strip()

    def _score_pair(self, pair):
        prediction, reference = pair
        prompt = json.dumps(
            {
                "prediction": prediction,
                "reference": reference,
            }
        )
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        response = completion.choices[0].message.content or ""
        return extract_numeric_score(response)

    def score(self, predictions, references):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(self.max_workers, 1)
        ) as executor:
            values = list(executor.map(self._score_pair, zip(predictions, references)))
        return score_mean_std(values)


class BartScorer:
    def __init__(self, model_name):
        hf_kwargs = huggingface_load_kwargs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **hf_kwargs,
        ).to(self.device)
        self.model.eval()

    def score(self, predictions, references, batch_size, max_length):
        scores = []
        for pred_batch, ref_batch in zip(
            batch_items(predictions, batch_size),
            batch_items(references, batch_size),
        ):
            encoder_inputs = self.tokenizer(
                pred_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            label_inputs = self.tokenizer(
                ref_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoder_inputs = {
                key: value.to(self.device) for key, value in encoder_inputs.items()
            }
            labels = label_inputs["input_ids"].to(self.device)
            label_mask = label_inputs["attention_mask"].to(self.device).bool()
            labels = labels.masked_fill(~label_mask, -100)

            with torch.no_grad():
                logits = self.model(**encoder_inputs, labels=labels).logits.float()

            valid_mask = labels.ne(-100)
            gather_labels = labels.masked_fill(~valid_mask, 0)
            token_log_probs = F.log_softmax(logits, dim=-1).gather(
                2, gather_labels.unsqueeze(-1)
            ).squeeze(-1)
            seq_scores = (
                (token_log_probs * valid_mask).sum(dim=1)
                / valid_mask.sum(dim=1).clamp_min(1)
            )
            scores.extend(seq_scores.detach().cpu().tolist())

        scores = np.asarray(scores)
        return float(np.mean(scores)), float(np.std(scores))


class LlamaScorer:
    def __init__(self, model_name, load_in_8bit=False):
        hf_kwargs = huggingface_load_kwargs()
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if load_in_8bit and torch.cuda.is_available():
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs, **hf_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = self.model.get_input_embeddings().weight.device
        self.model.eval()

    def score(self, predictions, references, batch_size, max_length):
        scores = []
        prompts = [f"Prediction: {prediction}\nReference:" for prediction in predictions]
        full_texts = [
            f"{prompt} {reference}" for prompt, reference in zip(prompts, references)
        ]

        for prompt_batch, full_batch in zip(
            batch_items(prompts, batch_size),
            batch_items(full_texts, batch_size),
        ):
            prompt_inputs = self.tokenizer(
                prompt_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            full_inputs = self.tokenizer(
                full_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
            full_inputs = {
                key: value.to(self.device) for key, value in full_inputs.items()
            }

            with torch.no_grad():
                logits = self.model(**full_inputs).logits.float()

            shift_logits = logits[:, :-1, :]
            shift_labels = full_inputs["input_ids"][:, 1:].to(logits.device)
            valid_mask = full_inputs["attention_mask"][:, 1:].to(logits.device).bool()

            for row_idx, prompt_length in enumerate(prompt_lengths):
                prompt_prefix = max(int(prompt_length) - 1, 0)
                if prompt_prefix > 0:
                    valid_mask[row_idx, :prompt_prefix] = False

            gather_labels = shift_labels.masked_fill(~valid_mask, 0)
            token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(
                2, gather_labels.unsqueeze(-1)
            ).squeeze(-1)
            seq_scores = (
                (token_log_probs * valid_mask).sum(dim=1)
                / valid_mask.sum(dim=1).clamp_min(1)
            )
            scores.extend(seq_scores.detach().cpu().tolist())

        scores = np.asarray(scores)
        return float(np.mean(scores)), float(np.std(scores))


class MetricScore:
    def __init__(self):
        self.results_path = args.results_path or (
            f"data/{args.dataset}/tst_results_{args.results_tag}.jsonl"
        )
        print(f"dataset: {args.dataset}")
        print(f"results_path: {self.results_path}")

        with open(self.results_path, "r") as file:
            self.records = [json.loads(line) for line in file if line.strip()]
        if args.max_samples is not None:
            self.records = self.records[: args.max_samples]

        metadata_override = load_metadata_override(args.metadata_path)
        if metadata_override:
            for record in self.records:
                iid = int(record["iid"])
                if iid in metadata_override:
                    record["item_metadata_text"] = metadata_override[iid]

        self.predictions = [record["prediction"] for record in self.records]
        self.references = [record["reference"] for record in self.records]

    def factual_precision(self):
        precision_scores = []
        matched_total = 0
        predicted_total = 0

        for record in self.records:
            predicted_attributes = extract_attribute_ngrams(record["prediction"])
            metadata_attributes = extract_attribute_ngrams(record["item_metadata_text"])
            if not predicted_attributes:
                precision_scores.append(1.0)
                continue
            matches = predicted_attributes & metadata_attributes
            matched_total += len(matches)
            predicted_total += len(predicted_attributes)
            precision_scores.append(len(matches) / len(predicted_attributes))

        overall_precision = matched_total / predicted_total if predicted_total else 1.0
        return (
            float(overall_precision),
            float(np.std(np.asarray(precision_scores))),
            matched_total,
            predicted_total,
        )

    def get_score(self):
        scores = {}
        (
            bert_precision,
            bert_recall,
            bert_f1,
            bert_precision_std,
            bert_recall_std,
            bert_f1_std,
        ) = bert_score(self.predictions, self.references)
        rouge1, rouge1_std = rouge_n_scores(self.predictions, self.references, 1)
        rouge2, rouge2_std = rouge_n_scores(self.predictions, self.references, 2)
        rougeL, rougeL_std = rouge_l_scores(self.predictions, self.references)
        bleu_scores = corpus_bleu_scores(self.predictions, self.references)
        avg_length = average_prediction_length(self.predictions)

        if args.skip_bart_score:
            bart_score_mean = None
            bart_score_std = None
        else:
            bart_score_mean, bart_score_std = BartScorer(args.bart_model_name).score(
                self.predictions,
                self.references,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

        if args.skip_llama_score:
            llama_score_mean = None
            llama_score_std = None
        else:
            llama_score_mean, llama_score_std = LlamaScorer(
                args.llama_model_name,
                load_in_8bit=args.llama_load_in_8bit,
            ).score(
                self.predictions,
                self.references,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

        if args.gpt_judge_model:
            api_key = os.environ.get(args.gpt_judge_api_key_env, "")
            base_url = args.gpt_judge_base_url or os.environ.get("OPENAI_BASE_URL", "")
            gpt_score_mean, gpt_score_std = GptJudgeScorer(
                model_name=args.gpt_judge_model,
                api_key=api_key,
                base_url=base_url,
                max_workers=args.gpt_judge_max_workers,
            ).score(self.predictions, self.references)
        else:
            gpt_score_mean = None
            gpt_score_std = None

        factual_precision_mean, factual_precision_std, matched_total, predicted_total = (
            self.factual_precision()
        )
        tokens_predict = [sentence.split() for sentence in self.predictions]
        usr, _ = unique_sentence_percent(tokens_predict)

        scores["bert_precision"] = bert_precision
        scores["bert_recall"] = bert_recall
        scores["bert_f1"] = bert_f1
        scores["rouge1"] = rouge1
        scores["rouge2"] = rouge2
        scores["rougeL"] = rougeL
        scores["bleu"] = bleu_scores["bleu"]
        scores["bleu1"] = bleu_scores["bleu1"]
        scores["bleu2"] = bleu_scores["bleu2"]
        scores["bleu3"] = bleu_scores["bleu3"]
        scores["bleu4"] = bleu_scores["bleu4"]
        scores["avg_length"] = avg_length
        scores["factual_precision"] = factual_precision_mean
        scores["matched_attributes"] = matched_total
        scores["predicted_attributes"] = predicted_total
        scores["usr"] = usr

        scores["bert_precision_std"] = bert_precision_std
        scores["bert_recall_std"] = bert_recall_std
        scores["bert_f1_std"] = bert_f1_std
        scores["rouge1_std"] = rouge1_std
        scores["rouge2_std"] = rouge2_std
        scores["rougeL_std"] = rougeL_std
        scores["factual_precision_std"] = factual_precision_std
        if bart_score_mean is not None:
            scores["bart_score"] = bart_score_mean
            scores["bart_score_std"] = bart_score_std
        if llama_score_mean is not None:
            scores["llama_score"] = llama_score_mean
            scores["llama_score_std"] = llama_score_std
        if gpt_score_mean is not None:
            scores["gpt_score"] = gpt_score_mean
            scores["gpt_score_std"] = gpt_score_std
        return scores

    def print_score(self):
        scores = self.get_score()
        print(f"dataset: {args.dataset}")
        print(f"results_tag: {args.results_tag}")
        print("Explainability Evaluation Metrics:")
        if "llama_score" in scores:
            print(f"llama_score: {scores['llama_score']:.4f}")
        if "gpt_score" in scores:
            print(f"gpt_score: {scores['gpt_score']:.4f}")
        if "bart_score" in scores:
            print(f"bart_score: {scores['bart_score']:.4f}")
        print(f"bert_precision: {scores['bert_precision']:.4f}")
        print(f"bert_recall: {scores['bert_recall']:.4f}")
        print(f"bert_f1: {scores['bert_f1']:.4f}")
        print(f"rouge1: {scores['rouge1']:.4f}")
        print(f"rouge2: {scores['rouge2']:.4f}")
        print(f"rougeL: {scores['rougeL']:.4f}")
        print(f"bleu: {scores['bleu']:.4f}")
        print(f"bleu1: {scores['bleu1']:.4f}")
        print(f"bleu2: {scores['bleu2']:.4f}")
        print(f"bleu3: {scores['bleu3']:.4f}")
        print(f"bleu4: {scores['bleu4']:.4f}")
        print(f"factual_precision: {scores['factual_precision']:.4f}")
        print(
            "attribute_matches: "
            f"{scores['matched_attributes']}/{scores['predicted_attributes']}"
        )
        print(f"usr: {scores['usr']:.4f}")
        print(f"avg_length: {scores['avg_length']:.2f}")
        print("-" * 30)
        print("Standard Deviation:")
        if "llama_score_std" in scores:
            print(f"llama_score_std: {scores['llama_score_std']:.4f}")
        if "gpt_score_std" in scores:
            print(f"gpt_score_std: {scores['gpt_score_std']:.4f}")
        if "bart_score_std" in scores:
            print(f"bart_score_std: {scores['bart_score_std']:.4f}")
        print(f"bert_precision_std: {scores['bert_precision_std']:.4f}")
        print(f"bert_recall_std: {scores['bert_recall_std']:.4f}")
        print(f"bert_f1_std: {scores['bert_f1_std']:.4f}")
        print(f"rouge1_std: {scores['rouge1_std']:.4f}")
        print(f"rouge2_std: {scores['rouge2_std']:.4f}")
        print(f"rougeL_std: {scores['rougeL_std']:.4f}")
        print(f"factual_precision_std: {scores['factual_precision_std']:.4f}")
