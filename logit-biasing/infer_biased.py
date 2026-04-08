import pickle
import torch
from explainer.models.explainer_mlp import XRecMLPExplainer


def run_inference(config: dict, user_id: int, item_id: int) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load vocab ─────────────────────────────────────────────────────────
    with open(config["vocab_path"], "rb") as f:
        vocab_data = pickle.load(f)
    item_vocab   = vocab_data["item_vocab"]        # {item_id_str -> set[int]}
    blacklist_ids = vocab_data["blacklist_ids"]

    verified_ids = item_vocab.get(str(item_id), set())

    # ── Load model ─────────────────────────────────────────────────────────
    model = XRecMLPExplainer(
        llm_model_name=config["llm_model_name"],
        collab_emb_dim=config["collab_emb_dim"],
        device=device,
    )
    model.load_state_dict(torch.load(config["output_model_path"], map_location=device))
    model.eval()

    # ── Load embeddings ────────────────────────────────────────────────────
    with open(config["user_emb_path"], "rb") as f:
        user_embs = torch.tensor(pickle.load(f), dtype=torch.float32).to(device)
    with open(config["item_emb_path"], "rb") as f:
        item_embs = torch.tensor(pickle.load(f), dtype=torch.float32).to(device)

    user_collab = user_embs[user_id].unsqueeze(0)     # (1, emb_dim)
    item_collab = item_embs[item_id].unsqueeze(0)     # (1, emb_dim)

    # ── Build prompt ───────────────────────────────────────────────────────
    prompt_text = (
        "Explain why this item is a good recommendation for the user "
        "based on their purchase history and item features.\n\nExplanation:"
    )
    encoded = model.tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )

    explanation = model.generate_explanation(
        user_collab_emb=user_collab,
        item_collab_emb=item_collab,
        prompt_input_ids=encoded["input_ids"].to(device),
        prompt_attention_mask=encoded["attention_mask"].to(device),
        verified_token_ids=verified_ids,
        blacklist_token_ids=blacklist_ids,
        positive_bias=config.get("positive_bias", 15.0),
        negative_bias=config.get("negative_bias", 10.0),
        max_new_tokens=config.get("max_new_tokens", 128),
    )
    return explanation


if __name__ == "__main__":
    config = {
        "llm_model_name":    "meta-llama/Llama-2-7b-hf",
        "collab_emb_dim":    64,
        "vocab_path":        "data/amazon/item_vocab.pkl",
        "user_emb_path":     "data/amazon/user_emb.pkl",
        "item_emb_path":     "data/amazon/item_emb.pkl",
        "output_model_path": "checkpoints/xrec_mlp.pt",
        "positive_bias":     15.0,
        "negative_bias":     10.0,
        "max_new_tokens":    128,
    }
    result = run_inference(config, user_id=42, item_id=101)
    print("Generated Explanation:\n", result)