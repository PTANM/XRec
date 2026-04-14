# coding: utf-8
# build_vocab.py

import json
import pickle
import nltk
from transformers import AutoTokenizer

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

MODEL_NAME  = "meta-llama/Llama-2-7b-hf"
DATA_PATH   = "/scratch/user/kiarab/XRec/data/amazon/item_profile.json"
OUTPUT_PATH = "/scratch/user/kiarab/XRec/data/amazon/item_vocab.pkl"

BLACKLIST_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "has", "have",
    "it", "this", "that", "be", "been", "being", "very", "really",
    "also", "just", "so", "but", "and", "or", "of", "in", "on",
    "at", "to", "for", "with", "as", "by", "from", "about",
]

# Keep only nouns and adjectives
KEEP_POS = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}


def extract_text_from_completion(completion: str) -> str:
    """Parse inner JSON and concatenate summarization + reasoning."""
    try:
        inner = json.loads(completion)
        summarization = inner.get("summarization", "")
        reasoning     = inner.get("reasoning", "")
        return f"{summarization} {reasoning}".strip()
    except (json.JSONDecodeError, TypeError):
        return completion


def extract_content_words(text: str) -> list:
    """Extract only nouns and adjectives using POS tagging."""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    content_words = [
        word.lower() for word, tag in tagged
        if tag in KEEP_POS and len(word) > 2  # skip very short words
    ]
    return list(set(content_words))


def build_item_vocab(data_path: str, tokenizer, output_path: str):
    item_vocab: dict = {}

    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record     = json.loads(line)
            item_id    = str(record["iid"])
            completion = record.get("completion", "")
            text       = extract_text_from_completion(completion)

            # Extract only content words
            content_words = extract_content_words(text)

            # Tokenize each content word
            token_ids: set = set()
            for word in content_words:
                ids = tokenizer.encode(word, add_special_tokens=False)
                token_ids.update(ids)

            item_vocab[item_id] = token_ids

    # Compute blacklist token IDs
    blacklist_ids: set = set()
    for word in BLACKLIST_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        ids += tokenizer.encode(word.capitalize(), add_special_tokens=False)
        blacklist_ids.update(ids)

    output = {"item_vocab": item_vocab, "blacklist_ids": blacklist_ids}
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    # Print some stats
    avg_vocab_size = sum(len(v) for v in item_vocab.values()) / len(item_vocab)
    print(f"Built vocab for {len(item_vocab)} items")
    print(f"Average tokens per item: {avg_vocab_size:.1f}")
    print(f"Blacklist size: {len(blacklist_ids)} tokens")
    print(f"Saved to {output_path}")
    return output


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    build_item_vocab(DATA_PATH, tokenizer, OUTPUT_PATH)