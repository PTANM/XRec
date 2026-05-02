import json
import pickle
import os
import nltk
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
BASE_DIR = "/scratch/user/kiarab/XRec/FCLB/data/amazon"  # Make as input

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

# Top-K content words to keep per item/user after TF-IDF ranking.
# Smaller = tighter, more discriminative bias targets.
TFIDF_TOP_K = 20


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


def tfidf_filter(words_per_doc: dict, top_k: int = TFIDF_TOP_K) -> dict:
    """
    Given {id: [word, ...]} dicts, return {id: [top_k_words]} where
    words are ranked by their TF-IDF score within each document.

    Using TF-IDF means high-frequency-across-all-items words (e.g.
    "product", "quality") get down-weighted, keeping only the words
    that are truly discriminative for a given item/user.
    """
    if not words_per_doc:
        return {}

    ids = list(words_per_doc.keys())
    # Each document is the space-joined word set for that id
    corpus = [" ".join(words_per_doc[i]) for i in ids]

    try:
        vec = TfidfVectorizer(max_features=2000)
        tfidf_matrix = vec.fit_transform(corpus)          # (N, V)
        feature_names = vec.get_feature_names_out()
    except ValueError:
        # Corpus too small / empty; fall back to raw word sets
        return words_per_doc

    filtered = {}
    for row_idx, doc_id in enumerate(ids):
        row = tfidf_matrix[row_idx]                       # sparse (1, V)
        scores = row.toarray().flatten()
        # Keep only words that actually appeared in this document
        nonzero = scores.nonzero()[0]
        if len(nonzero) == 0:
            filtered[doc_id] = words_per_doc[doc_id]
            continue
        # Sort by descending TF-IDF score, keep top_k
        top_indices = nonzero[scores[nonzero].argsort()[::-1][:top_k]]
        filtered[doc_id] = list(feature_names[top_indices])

    return filtered


def words_to_token_ids(words, tokenizer):
    """
    FIX: Use a leading space when encoding so we match the subword token
    that actually appears mid-sentence in LLaMA (e.g. ' cat' not 'cat').
    Only keep words that encode as a SINGLE token to avoid false-positive
    bias on unrelated subword fragments.
    """
    token_ids = set()
    for word in words:
        for variant in (word, word.capitalize()):
            # Leading space = word appears after a space in context
            ids = tokenizer.encode(" " + variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])
            # Also try without leading space (e.g. sentence-initial)
            ids_no_space = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids_no_space) == 1:
                token_ids.add(ids_no_space[0])
    return token_ids


def build_item_vocab(data_path, tokenizer):
    raw_words = {}   # {iid: [word, ...]} before TF-IDF
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            item_id = str(record["iid"])
            completion = record.get("completion", "")
            text = extract_text_from_completion(completion)
            raw_words[item_id] = extract_content_words(text)

    print(f"  TF-IDF filtering {len(raw_words)} items (top {TFIDF_TOP_K} words each)...")
    filtered_words = tfidf_filter(raw_words, top_k=TFIDF_TOP_K)

    item_vocab = {}
    item_words = {}
    for item_id, words in filtered_words.items():
        token_ids = words_to_token_ids(words, tokenizer)
        item_vocab[item_id] = token_ids
        item_words[item_id] = set(words)

    return item_vocab, item_words


def build_user_vocab(data_path, tokenizer):
    user_vocab = {}
    user_words = {}
    if not os.path.exists(data_path):
        print(f"WARNING: {data_path} not found. User vocab will be empty.")
        return user_vocab, user_words

    raw_words = {}
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user_id = str(record.get("uid", record.get("user_id", "")))
            completion = record.get("completion", "")
            text = extract_text_from_completion(completion)
            raw_words[user_id] = extract_content_words(text)

    print(f"  TF-IDF filtering {len(raw_words)} users (top {TFIDF_TOP_K} words each)...")
    filtered_words = tfidf_filter(raw_words, top_k=TFIDF_TOP_K)

    for user_id, words in filtered_words.items():
        token_ids = words_to_token_ids(words, tokenizer)
        user_vocab[user_id] = token_ids
        user_words[user_id] = set(words)

    return user_vocab, user_words


def build_blacklist_ids(tokenizer):
    """
    FIX: apply the same leading-space + single-token filter to the
    blacklist so we don't accidentally blacklist subword fragments.
    """
    blacklist_ids = set()
    for word in BLACKLIST_WORDS:
        for variant in (word, word.capitalize()):
            ids = tokenizer.encode(" " + variant, add_special_tokens=False)
            if len(ids) == 1:
                blacklist_ids.add(ids[0])
            ids2 = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids2) == 1:
                blacklist_ids.add(ids2[0])
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
