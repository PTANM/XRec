import pickle
import os

VOCAB_PATH = "/scratch/user/kiarab/XRec/data/amazon/item_vocab.pkl"

# 1. Check file exists and is non-trivial
size_kb = os.path.getsize(VOCAB_PATH) / 1024
print(f"File size: {size_kb:.1f} KB")
assert size_kb > 1, "ERROR: File is suspiciously small"

# 2. Load and check structure
with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)

print(f"\nKeys in vocab_data: {list(vocab_data.keys())}")
assert "item_vocab" in vocab_data, "ERROR: missing item_vocab key"
assert "blacklist_ids" in vocab_data, "ERROR: missing blacklist_ids key"

item_vocab = vocab_data["item_vocab"]
blacklist_ids = vocab_data["blacklist_ids"]

# 3. Check item_vocab is non-empty and has correct types
print(f"\nTotal items: {len(item_vocab)}")
assert len(item_vocab) > 0, "ERROR: item_vocab is empty"

# 4. Spot-check a few entries
sample_keys = list(item_vocab.keys())[:5]
for key in sample_keys:
    ids = item_vocab[key]
    print(f"  Item '{key}': {len(ids)} token IDs, type={type(ids).__name__}")
    assert isinstance(ids, set), f"ERROR: expected set, got {type(ids)}"
    assert all(isinstance(x, int) for x in ids), "ERROR: non-int token ID found"

# 5. Check for degenerate cases
sizes = [len(v) for v in item_vocab.values()]
avg_size = sum(sizes) / len(sizes)
empty_count = sum(1 for s in sizes if s == 0)
tiny_count = sum(1 for s in sizes if 0 < s < 3)
huge_count = sum(1 for s in sizes if s > 500)

print(f"\nToken ID stats per item:")
print(f"  avg:   {avg_size:.1f}")
print(f"  min:   {min(sizes)}")
print(f"  max:   {max(sizes)}")
print(f"  empty: {empty_count} items (0 tokens)")
print(f"  tiny:  {tiny_count} items (<3 tokens)")
print(f"  huge:  {huge_count} items (>500 tokens)")

if empty_count > len(item_vocab) * 0.1:
    print("WARNING: >10% of items have zero tokens -- check item_profile.json parsing")

# 6. Check blacklist
print(f"\nBlacklist: {len(blacklist_ids)} token IDs, type={type(blacklist_ids).__name__}")
assert len(blacklist_ids) > 0, "ERROR: blacklist is empty"

# 7. Check for overlap between verified and blacklist
overlap_counts = []
for iid, ids in item_vocab.items():
    overlap = ids & blacklist_ids
    overlap_counts.append(len(overlap))
avg_overlap = sum(overlap_counts) / len(overlap_counts)
print(f"\nAvg overlap (item tokens and blacklist): {avg_overlap:.1f}")
if avg_overlap > 5:
    print("WARNING: High overlap means blacklist may suppress item-relevant tokens")

# 8. Verify token IDs are in valid LLaMA-2 range (vocab size = 32000)
LLAMA_VOCAB = 32000
all_ids = set()
for ids in item_vocab.values():
    all_ids.update(ids)
out_of_range = [x for x in all_ids if x >= LLAMA_VOCAB]
print(f"\nUnique token IDs across all items: {len(all_ids)}")
print(f"Out-of-range (>= {LLAMA_VOCAB}): {len(out_of_range)}")

if not out_of_range and empty_count == 0:
    print("\nAll checks passed!")
else:
    print("\nReview warnings above")