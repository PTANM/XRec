import json
import pickle
import os
import nltk
from transformers import AutoTokenizer
from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
BASE_DIR = "/scratch/user/kiarab/XRec/data/amazon"  # Make as input

ITEM_PROFILE_PATH = os.path.join(BASE_DIR, "item_profile.json")
USER_PROFILE_PATH = os.path.join(BASE_DIR, "user_profile.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "item_vocab.pkl")

BLACKLIST_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "has", "have",
    "it", "this", "that", "be", "been", "being", "very", "really",
    "also", "just", "so", "but", "and", "or", "of", "in", "on",
    "at", "to", "for", "with", "as", "by", "from", "about",
]

KEEP_POS = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS'}


def extract_text_from_completion(completion: str) -> str:
    try:
        inner = json.loads(completion)
        summarization = inner.get("summarization", "")
        reasoning = inner.get("reasoning", "")
        return f"{summarization} {reasoning}".strip()
    except (json.JSONDecodeError, TypeError):
        return completion


def extract_content_words(text: str) -> list:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    content_words = [
        word.lower() for word, tag in tagged
        if tag in KEEP_POS and len(word) > 2
    ]
    return list(set(content_words))


def words_to_token_ids(words, tokenizer):
    token_ids = set()
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        token_ids.update(ids)
    return token_ids


def build_item_vocab(data_path, tokenizer):
    item_vocab = {}
    item_words = {}
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            item_id = str(record["iid"])
            completion = record.get("completion", "")
            text = extract_text_from_completion(completion)
            content_words = extract_content_words(text)
            token_ids = words_to_token_ids(content_words, tokenizer)
            item_vocab[item_id] = token_ids
            item_words[item_id] = set(content_words)
    return item_vocab, item_words


def build_user_vocab(data_path, tokenizer):
    user_vocab = {}
    user_words = {}
    if not os.path.exists(data_path):
        print(f"WARNING: {data_path} not found. User vocab will be empty.")
        return user_vocab, user_words
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user_id = str(record.get("uid", record.get("user_id", "")))
            completion = record.get("completion", "")
            text = extract_text_from_completion(completion)
            content_words = extract_content_words(text)
            token_ids = words_to_token_ids(content_words, tokenizer)
            user_vocab[user_id] = token_ids
            user_words[user_id] = set(content_words)
    return user_vocab, user_words


def build_blacklist_ids(tokenizer):
    blacklist_ids = set()
    for word in BLACKLIST_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        ids += tokenizer.encode(word.capitalize(), add_special_tokens=False)
        blacklist_ids.update(ids)
    return blacklist_ids


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print("Building item vocab...")
    item_vocab, item_words = build_item_vocab(ITEM_PROFILE_PATH, tokenizer)

    print("Building user vocab...")
    user_vocab, user_words = build_user_vocab(USER_PROFILE_PATH, tokenizer)

    blacklist_ids = build_blacklist_ids(tokenizer)

    output = {
        "item_vocab": item_vocab,
        "item_words": item_words,
        "user_vocab": user_vocab,
        "user_words": user_words,
        "blacklist_ids": blacklist_ids,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)

    avg_item = sum(len(v) for v in item_vocab.values()) / max(len(item_vocab), 1)
    avg_user = sum(len(v) for v in user_vocab.values()) / max(len(user_vocab), 1)
    print(f"Items: {len(item_vocab)}, avg tokens/item: {avg_item:.1f}")
    print(f"Users: {len(user_vocab)}, avg tokens/user: {avg_user:.1f}")
    print(f"Blacklist: {len(blacklist_ids)} token IDs")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()