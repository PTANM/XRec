"""
Post FIne Tuning Explanation Generation Script

Usage:
    # Full dataset
    python generate_cf_explanations.py --lora_dir outputs/cf_lora/epoch_3

    # Custom output path
    python generate_cf_explanations.py --lora_dir outputs/cf_lora/epoch_3 \
        --output counterfactual_finetuned_epoch3.json

"""

import os
import sys
import json
import pickle
import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT     = Path("./")
DATASET  = "amazon"
DATA_DIR = ROOT / "data" / DATASET
if not DATA_DIR.exists():
    DATA_DIR = ROOT / DATASET
CF_XLSX  = ROOT / "counterfactual_training_dataset_evaluated.csv"
CF_JSON  = ROOT / "counterfactual_training_dataset.json"


def load_lora_weights(llama_model, lora_dir: Path, device):
    """Inject LoRA structure and load saved delta weights."""
    # Import LoRA helpers from finetune_cf
    sys.path.insert(0, str(ROOT))
    from finetune_cf import LoRALinear, inject_lora, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS

    n = inject_lora(llama_model, LORA_TARGETS, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    print(f"  LoRA injected into {n} layers")

    lora_path = lora_dir / "lora_weights.pt"
    state = torch.load(lora_path, map_location=device)

    for name, m in llama_model.named_modules():
        if isinstance(m, LoRALinear):
            key_A = f"{name}.lora_A"
            key_B = f"{name}.lora_B"
            if key_A in state:
                m.lora_A.weight.data = state[key_A].to(device)
            if key_B in state:
                m.lora_B.weight.data = state[key_B].to(device)

    print(f"  LoRA weights loaded from {lora_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", type=str, default="outputs/cf_lora/epoch_3",
                        help="Checkpoint directory with lora_weights.pt, user_converter.pkl, item_converter.pkl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records (for quick testing). Default: all records.")
    parser.add_argument("--output", type=str, default="counterfactual_finetuned_results.json",
                        help="Output JSON path")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N records")
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    assert lora_dir.exists(), f"Checkpoint dir not found: {lora_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {lora_dir}")
    print(f"Output: {args.output}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    with open(CF_JSON) as f:
        json_records = json.load(f)
    print(f"  JSON records: {len(json_records)}")

    # Load csv for chatgpt_response reference
    df_xlsx = pd.read_csv(CF_XLSX)
    chatgpt_by_uid = {int(row["user_id"]): str(row.get("chatgpt_response", ""))
                      for _, row in df_xlsx.iterrows()}
    judgement_by_uid = {int(row["user_id"]): str(row.get("judgement_of_llama_explanation", ""))
                        for _, row in df_xlsx.iterrows()}
    print(f"  csv rows: {len(df_xlsx)}")

    # Only keep records with ChatGPT reference — 1777 evaluated records
    evaluated_uids = {uid for uid, ref in chatgpt_by_uid.items() if ref.strip()}
    records = [r for r in json_records if r["user_id"] in evaluated_uids]
    print(f"  Records with ChatGPT reference: {len(records)} / {len(json_records)} total")
    if args.limit:
        records = records[:args.limit]
    print(f"  Generating for: {len(records)} records")

    # ── Load embeddings ────────────────────────────────────────────────────────
    print("\n[2/4] Loading embeddings...")
    with open(DATA_DIR / "user_emb.pkl", "rb") as f:
        user_emb = pickle.load(f).cpu().float()
    with open(DATA_DIR / "item_emb.pkl", "rb") as f:
        item_emb = pickle.load(f).cpu().float()
    print(f"  user_emb: {user_emb.shape}, item_emb: {item_emb.shape}")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[3/4] Loading fine-tuned model...")
    explainer_path = str(ROOT / "explainer")
    if explainer_path not in sys.path:
        sys.path.insert(0, explainer_path)
    for key in list(sys.modules.keys()):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]
    importlib.invalidate_caches()

    from models.explainer import Explainer
    model = Explainer()

    # Load fine-tuned MoE adapter weights
    model.user_embedding_converter.load_state_dict(
        torch.load(lora_dir / "user_converter.pkl", map_location=device)
    )
    model.item_embedding_converter.load_state_dict(
        torch.load(lora_dir / "item_converter.pkl", map_location=device)
    )

    # Load LoRA delta weights on top of the frozen LLaMA backbone
    load_lora_weights(model.model, lora_dir, device)

    model.to(device)
    model.eval()
    print("  Model loaded with fine-tuned weights")

    # ── Generate explanations (with checkpoint/resume) ─────────────────────────
    print(f"\n[4/4] Generating explanations...")
    output_path = Path(args.output)

    # Resume: load partial results if output already exists
    results   = []
    start_idx = 0
    if output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"  Resuming from record {start_idx} ({start_idx} already done)")
    else:
        print(f"  Starting fresh (0 done)")

    errors = 0

    with torch.no_grad():
        for i, r in enumerate(tqdm(records[start_idx:], desc="Generating",
                                   initial=start_idx, total=len(records))):
            uid   = min(r["user_id"], user_emb.shape[0] - 1)
            iid   = min(r["target_item_id"], item_emb.shape[0] - 1)
            bids  = [min(b, item_emb.shape[0] - 1) for b in r["bridge_item_ids"]]
            alpha = float(r["minimum_alpha"])

            user_vec   = user_emb[uid]
            bridge_vec = item_emb[bids].mean(dim=0)
            perturbed  = (user_vec + alpha * (bridge_vec - user_vec)).unsqueeze(0).to(device)
            item_vec   = item_emb[iid].unsqueeze(0).to(device)

            prompt = r["counterfactual_prompt"]

            try:
                generated = model.generate(perturbed, item_vec, [prompt])
                explanation = generated[0].strip() if generated else ""
            except Exception as e:
                explanation = ""
                errors += 1

            results.append({
                "user_id":                  r["user_id"],
                "target_name":              r["target_name"],
                "target_profile":           r["target_profile"],
                "bridge_names":             r.get("bridge_names", []),
                "bridge_profiles":          r.get("bridge_profiles", []),
                "minimum_alpha":            r["minimum_alpha"],
                "original_score":           r.get("original_score"),
                "final_score":              r.get("final_score"),
                "recommendation_threshold": r.get("recommendation_threshold"),
                "gap_to_to_threshold":      r.get("gap_to_to_threshold"),
                "final_rank":               r.get("final_rank"),
                "strategy":                 r.get("strategy", ""),
                "counterfactual_prompt":    prompt,
                "llama_explanation":        explanation,
                "chatgpt_reference":        chatgpt_by_uid.get(r["user_id"], ""),
                "judgement":                judgement_by_uid.get(r["user_id"], ""),
            })

            if (i + 1) % args.save_every == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Checkpoint saved: {len(results)} records done")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results → {output_path}")
    if errors:
        print(f"  {errors} records had generation errors (empty explanation)")

    # Save as tst_pred.pkl / tst_ref.pkl for evaluation/main.py
    predictions = [r["llama_explanation"] for r in results]
    references  = [r["chatgpt_reference"]  for r in results]
    with open(DATA_DIR / "tst_pred.pkl", "wb") as f:
        pickle.dump(predictions, f)
    with open(DATA_DIR / "tst_ref.pkl", "wb") as f:
        pickle.dump(references, f)
    print(f"tst_pred.pkl and tst_ref.pkl saved → {DATA_DIR}")

    sample = next((r for r in results if r["llama_explanation"]), results[0])
    print(f"\nSample record:")
    print(f"  user_id:     {sample['user_id']}")
    print(f"  target:      {sample['target_name']}")
    print(f"  explanation: {sample['llama_explanation'][:200]}")
    print(f"  reference:   {sample['chatgpt_reference'][:200]}")



if __name__ == "__main__":
    main()
