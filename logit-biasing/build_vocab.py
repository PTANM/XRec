import json
import pickle
from transformers import AutoTokenizer

# ── config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "meta-llama/Llama-2-7b-hf"   # same tokenizer as XRec's LLM
DATA_PATH    = "data/amazon/data.json"        # XRec item profile JSON
OUTPUT_PATH  = "data/amazon/item_vocab.pkl"   # {item_id -> set of token IDs}

# Generic filler words to suppress during generation
BLACKLIST_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "has", "have",
    "it", "this", "that", "be", "been", "being", "very", "really",
    "also", "just", "so", "but", "and", "or", "of", "in", "on",
    "at", "to", "for", "with", "as", "by", "from", "about",
]

METADATA_KEYS = ["category", "brand", "price_range", "features", "tags"]
# ────────────────────────────────────────────────────────────────────────────

def extract_item_attributes(item_profile: dict) -> list[str]:
    """Pull discrete string attributes from one item's profile."""
    attributes = []
    for key in METADATA_KEYS:
        val = item_profile.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            attributes.extend([str(v).strip() for v in val if v])
        elif isinstance(val, str):
            # Split multi-word phrases into individual tokens too
            attributes.append(val.strip())
            attributes.extend(val.strip().split())
    return list(set(a for a in attributes if a))


def build_item_vocab(data_path: str, tokenizer, output_path: str):
    with open(data_path, "r") as f:
        data = json.load(f)  # list of {item_id, item_profile, ...}

    item_vocab: dict[str, set[int]] = {}

    for record in data:
        item_id      = str(record["item_id"])
        item_profile = record.get("item_profile", {})
        attributes   = extract_item_attributes(item_profile)

        token_ids: set[int] = set()
        for attr in attributes:
            # Encode without BOS; add_special_tokens=False keeps it clean
            ids = tokenizer.encode(attr, add_special_tokens=False)
            token_ids.update(ids)

        item_vocab[item_id] = token_ids

    # Also compute blacklist token IDs (shared across all items)
    blacklist_ids: set[int] = set()
    for word in BLACKLIST_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        # Also encode capitalised variant
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