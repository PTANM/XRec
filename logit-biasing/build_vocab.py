# build_vocab.py

import json
import pickle
from transformers import AutoTokenizer

MODEL_NAME   = "meta-llama/Llama-2-7b-hf"
DATA_PATH    = "../data/amazon/item_profile.json"
OUTPUT_PATH  = "../data/amazon/item_vocab.pkl"

BLACKLIST_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "has", "have",
    "it", "this", "that", "be", "been", "being", "very", "really",
    "also", "just", "so", "but", "and", "or", "of", "in", "on",
    "at", "to", "for", "with", "as", "by", "from", "about",
]


def extract_text_from_completion(completion: str) -> str:
    """Parse the inner JSON and concatenate summarization + reasoning."""
    try:
        inner = json.loads(completion)
        summarization = inner.get("summarization", "")
        reasoning     = inner.get("reasoning", "")
        return f"{summarization} {reasoning}".strip()
    except (json.JSONDecodeError, TypeError):
        # Fall back to raw string if parsing fails
        return completion


def build_item_vocab(data_path: str, tokenizer, output_path: str):
    item_vocab: dict[str, set[int]] = {}

    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record     = json.loads(line)
            item_id    = str(record["iid"])
            completion = record.get("completion", "")
            text       = extract_text_from_completion(completion)

            token_ids = set(tokenizer.encode(text, add_special_tokens=False))
            item_vocab[item_id] = token_ids

    # Compute blacklist token IDs
    blacklist_ids: set[int] = set()
    for word in BLACKLIST_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        ids += tokenizer.encode(word.capitalize(), add_special_tokens=False)
        blacklist_ids.update(ids)

    output = {"item_vocab": item_vocab, "blacklist_ids": blacklist_ids}
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Built vocab for {len(item_vocab)} items → {output_path}")
    return output


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    build_item_vocab(DATA_PATH, tokenizer, OUTPUT_PATH)