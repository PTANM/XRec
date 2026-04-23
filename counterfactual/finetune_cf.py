"""
finetune_cf.py
LoRA fine-tuning of XRec LLaMA explainer on counterfactual data.

What this script does:
1. Loads the counterfactual training dataset (JSON with prompts + explanations)
2. Loads user and item embeddings (pkl files)
3. Recomputes perturbed user vectors from stored alpha + bridge_ids
4. Adds LoRA adapters to LLaMA attention layers (q_proj, v_proj)
5. Keeps MoE adapters (user_embedding_converter, item_embedding_converter) trainable
6. Trains with cross-entropy loss only on the explanation tokens (after <EXPLAIN_POS>)
7. Saves LoRA weights + MoE adapter weights after each epoch

Run:
    python finetune_cf.py
    or:
    sbatch run_finetune.sh
"""

import os
import sys
import pickle
import importlib
import traceback
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT          = Path("./")                                  # working dir where GCD.py was run
DATASET       = "amazon"
DATA_DIR      = ROOT / DATASET           # data is at ./amazon/ on cluster (not ./data/amazon/)
CF_DATA_PATH  = ROOT / "counterfactual_training_dataset_evaluated.csv"   # evaluated csv with chatgpt_response

OUTPUT_DIR    = ROOT / "outputs" / "cf_lora"
LOG_FILE      = ROOT / "finetune_cf.log"

# Training hypers
BATCH_SIZE    = 4
GRAD_ACCUM    = 4          # effective batch size = BATCH_SIZE * GRAD_ACCUM = 16
LR            = 2e-4
EPOCHS        = 3
MAX_LENGTH    = 512        # max prompt+explanation token length
SAVE_EVERY    = 200        # save checkpoint every N steps

# LoRA config
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "v_proj"]   # LLaMA attention layers to add LoRA to

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Dataset ───────────────────────────────────────────────────────────────────

class CounterfactualDataset(Dataset):
    """
    Merges the evaluated xlsx (chatgpt_response + judgement) with the original
    JSON (bridge_item_ids, target_item_id, minimum_alpha) on user_id.

    Each record provides:
    - perturbed user vector recomputed from alpha + bridge_ids (same as original)
    - target item vector from item_emb[target_item_id]
    - full prompt (counterfactual_prompt) with <EXPLAIN_POS> marker
    - target explanation: chatgpt_response (high-quality gold standard)

    Only rows with judgement "Correct" or "Partially Correct" are kept.
    """
    def __init__(self, df: pd.DataFrame, json_records: list, user_emb: torch.Tensor, item_emb: torch.Tensor):
        self.user_emb = user_emb.float()
        self.item_emb = item_emb.float()

        # Build lookup from user_id -> json record (for bridge info)
        json_by_uid = {r["user_id"]: r for r in json_records}

        valid_judgements = {"Correct", "Partially Correct"}
        self.records = []
        skipped = 0

        for _, row in df.iterrows():
            exp       = str(row.get("chatgpt_response", ""))
            judgement = str(row.get("judgement_of_llama_explanation", ""))
            uid       = int(row["user_id"])

            if judgement not in valid_judgements:
                skipped += 1
                continue
            if len(exp.split()) < 10:
                skipped += 1
                continue
            if pd.isna(row.get("counterfactual_prompt")):
                skipped += 1
                continue

            jr = json_by_uid.get(uid)
            if jr is None or not jr.get("bridge_item_ids"):
                skipped += 1
                continue

            self.records.append({
                "user_id":       uid,
                "target_item_id": jr["target_item_id"],
                "bridge_item_ids": jr["bridge_item_ids"],
                "minimum_alpha":  float(jr["minimum_alpha"]),
                "prompt":        str(row["counterfactual_prompt"]),
                "explanation":   exp,
            })

        log(f"Dataset: {len(self.records)} valid records, {skipped} skipped")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r     = self.records[idx]
        uid   = min(r["user_id"], self.user_emb.shape[0] - 1)
        iid   = min(r["target_item_id"], self.item_emb.shape[0] - 1)
        bids  = [min(b, self.item_emb.shape[0] - 1) for b in r["bridge_item_ids"]]
        alpha = r["minimum_alpha"]

        # Recompute perturbed user vector (same as original training code)
        user_vec   = self.user_emb[uid]
        bridge_vec = self.item_emb[bids].mean(dim=0)
        perturbed  = user_vec + alpha * (bridge_vec - user_vec)

        return {
            "user_vector": perturbed,               # [64] perturbed toward bridge
            "item_vector": self.item_emb[iid],      # [64] target item
            "prompt":      r["prompt"],
            "explanation": r["explanation"],         # chatgpt_response gold standard
        }


def collate_fn(batch):
    """Stack vectors; keep text as lists for tokenization in the model."""
    return {
        "user_vector": torch.stack([b["user_vector"] for b in batch]),   # [B, 64]
        "item_vector": torch.stack([b["item_vector"] for b in batch]),   # [B, 64]
        "prompt":      [b["prompt"]      for b in batch],
        "explanation": [b["explanation"] for b in batch],
    }

# ── LoRA Layer ─────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adapters.
    Only the A and B matrices are trained; the original weight is frozen.
    """
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.linear   = linear
        self.lora_A   = nn.Linear(linear.in_features,  r, bias=False)
        self.lora_B   = nn.Linear(r, linear.out_features, bias=False)
        self.scaling  = alpha / r
        self.dropout  = nn.Dropout(dropout)

        # Initialise: A ~ N(0, 0.02), B = 0
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base   = self.linear(x)
        # Cast x to LoRA weight dtype (float32) to avoid float16/float32 mismatch
        x_cast = x.to(self.lora_A.weight.dtype)
        lora   = self.lora_B(self.lora_A(self.dropout(x_cast))) * self.scaling
        return base + lora.to(base.dtype)


def inject_lora(llama_model, target_modules, r, alpha, dropout):
    """
    Walk the LlamaForCausalLM and replace every nn.Linear whose name
    ends with any of `target_modules` with a LoRALinear wrapper.
    Returns the count of injected layers.
    """
    injected = 0
    for module_name, module in llama_model.named_modules():
        for attr_name in dir(module):
            if not attr_name.endswith(tuple(target_modules)):
                continue
            child = getattr(module, attr_name, None)
            if not isinstance(child, nn.Linear):
                continue
            setattr(module, attr_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            injected += 1
    return injected


def get_lora_params(llama_model):
    """Return only LoRA parameters for the optimiser."""
    params = []
    for m in llama_model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A.weight, m.lora_B.weight])
    return params


def save_lora_weights(llama_model, path: Path):
    """Save only LoRA delta weights (small file)."""
    state = {}
    for name, m in llama_model.named_modules():
        if isinstance(m, LoRALinear):
            state[f"{name}.lora_A"] = m.lora_A.weight.data.cpu()
            state[f"{name}.lora_B"] = m.lora_B.weight.data.cpu()
    torch.save(state, path)
    log(f"  LoRA weights saved → {path}  ({len(state)//2} pairs)")

# ── Training Step ─────────────────────────────────────────────────────────────

def training_step(model, batch, device):
    """
    Forward pass producing loss on explanation tokens only.

    The prompt is the part before <EXPLAIN_POS>.
    The target is (prompt + " " + explanation), with prompt tokens masked to -100.
    """
    tokenizer = model.tokenizer
    llama     = model.model

    user_vec = batch["user_vector"].to(device).float()   # [B, 64]
    item_vec = batch["item_vector"].to(device).float()   # [B, 64]

    # Convert embeddings through MoE adapters
    user_emb_converted = model.user_embedding_converter(user_vec).half()   # [B, 4096]
    item_emb_converted = model.item_embedding_converter(item_vec).half()   # [B, 4096]

    # Build full training sequence: prompt + space + explanation
    full_texts = [
        p.replace("<EXPLAIN_POS>", "") + " " + e
        for p, e in zip(batch["prompt"], batch["explanation"])
    ]
    prompt_only = [
        p.replace("<EXPLAIN_POS>", "")
        for p in batch["prompt"]
    ]

    # Tokenize full sequence
    enc_full = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    # Tokenize prompt only to find split position (no padding — returns list of lists)
    enc_prompt = tokenizer(
        prompt_only,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None,
    )

    # Build label tensor: -100 for prompt tokens, real ids for explanation tokens
    labels = enc_full["input_ids"].clone()
    for i, prompt_ids in enumerate(enc_prompt["input_ids"]):
        prompt_len = len(prompt_ids)   # list length, not tensor shape
        labels[i, :prompt_len] = -100
    # Also mask padding
    labels[enc_full["attention_mask"] == 0] = -100

    # Get input embeddings
    inputs_embeds = llama.get_input_embeddings()(enc_full["input_ids"])  # [B, L, 4096]

    # Inject user/item embeddings at <USER_EMBED> / <ITEM_EMBED> positions
    user_token_id = tokenizer.convert_tokens_to_ids("<USER_EMBED>")
    item_token_id = tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")

    ids_cpu = enc_full["input_ids"].cpu()
    user_positions = (ids_cpu == user_token_id).nonzero(as_tuple=True)
    item_positions = (ids_cpu == item_token_id).nonzero(as_tuple=True)

    if len(user_positions[0]) > 0:
        for bi, pos in zip(user_positions[0], user_positions[1]):
            inputs_embeds[bi, pos, :] = user_emb_converted[bi]
    if len(item_positions[0]) > 0:
        for bi, pos in zip(item_positions[0], item_positions[1]):
            inputs_embeds[bi, pos, :] = item_emb_converted[bi]

    # Forward pass through LLaMA (use_cache=False disables StaticCache during training)
    outputs = llama(
        inputs_embeds=inputs_embeds,
        attention_mask=enc_full["attention_mask"],
        user_embed=user_emb_converted,
        item_embed=item_emb_converted,
        user_embed_pos=torch.zeros(user_vec.shape[0], 1, dtype=torch.long, device=device),
        item_embed_pos=torch.zeros(user_vec.shape[0], 1, dtype=torch.long, device=device),
        use_cache=False,
    )

    # Compute cross-entropy loss on explanation tokens only
    logits       = outputs.logits                        # [B, L, V]
    shift_logits = logits[:, :-1, :].contiguous()        # [B, L-1, V]
    shift_labels = labels[:, 1:].contiguous()            # [B, L-1]

    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import huggingface_hub

    # ── Parse args ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Smoke test: use only 20 records, 1 epoch, save to outputs/cf_lora_test/")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest epoch checkpoint in OUTPUT_DIR")
    cli_args = parser.parse_args()

    # Override config for test mode
    if cli_args.test:
        EPOCHS     = 1
        SAVE_EVERY = 5     # checkpoint more frequently in test
        OUTPUT_DIR = ROOT / "outputs" / "cf_lora_test"
        log("⚡ TEST MODE: 20 records, 1 epoch")

    # HF login (offline is fine on Grace)
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        try:
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    log(f"Loading evaluated dataset from {CF_DATA_PATH} ...")
    # Try csv first, fallback to xlsx
    try:
        if CF_DATA_PATH.suffix == '.csv':
            df = pd.read_csv(CF_DATA_PATH)
        else:
            df = pd.read_excel(CF_DATA_PATH, engine='xlrd')
    except Exception as e:
        log(f"⚠️ Read failed: {e}")
        log("Falling back to csv...")
        csv_path = CF_DATA_PATH.with_suffix('.csv')
        df = pd.read_csv(csv_path)
    log(f"Total raw rows: {len(df)}")
    log(f"Judgement distribution:\n{df['judgement_of_llama_explanation'].value_counts().to_string()}")

    # Load original JSON for bridge_item_ids, target_item_id, minimum_alpha
    json_path = ROOT / "counterfactual_training_dataset.json"
    log(f"Loading JSON metadata from {json_path} ...")
    import json
    with open(json_path) as f:
        json_records = json.load(f)
    log(f"JSON records: {len(json_records)}")

    log("Loading embeddings ...")
    user_emb = load_pickle(DATA_DIR / "user_emb.pkl").cpu()
    item_emb = load_pickle(DATA_DIR / "item_emb.pkl").cpu()
    log(f"User embeddings: {user_emb.shape}, Item embeddings: {item_emb.shape}")

    dataset = CounterfactualDataset(df, json_records, user_emb, item_emb)

    # Limit to 20 records in test mode
    if cli_args.test:
        dataset.records = dataset.records[:20]
        log(f"⚡ TEST MODE: trimmed dataset to {len(dataset.records)} records")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)

    # ── Load XRec explainer model ──────────────────────────────────────────────
    log("Loading XRec Explainer model ...")
    explainer_path = str(ROOT / "explainer")
    if explainer_path not in sys.path:
        sys.path.insert(0, explainer_path)
    for key in list(sys.modules.keys()):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]
    importlib.invalidate_caches()

    from models.explainer import Explainer
    model = Explainer()
    model.user_embedding_converter.load_state_dict(
        torch.load(DATA_DIR / "user_converter.pkl", map_location=device)
    )
    model.item_embedding_converter.load_state_dict(
        torch.load(DATA_DIR / "item_converter.pkl", map_location=device)
    )
    log("Model loaded.")

    # ── Inject LoRA into LLaMA ─────────────────────────────────────────────────
    log(f"Injecting LoRA (r={LORA_R}, alpha={LORA_ALPHA}) into {LORA_TARGETS} ...")
    n_injected = inject_lora(model.model, LORA_TARGETS, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    log(f"Injected LoRA into {n_injected} layers.")

    # Move to device AFTER LoRA injection so all new lora_A/lora_B are on GPU
    model.to(device)
    log(f"Model moved to {device}.")

    # ── Set trainable parameters ───────────────────────────────────────────────
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze: (1) LoRA params, (2) MoE adapters
    lora_params = get_lora_params(model.model)
    moe_params  = list(model.user_embedding_converter.parameters()) + \
                  list(model.item_embedding_converter.parameters())

    for p in lora_params + moe_params:
        p.requires_grad = True

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100*trainable_params/total_params:.2f}%)")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01,
    )

    total_steps = EPOCHS * len(dataloader) // GRAD_ACCUM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Output directory ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    if cli_args.resume:
        epoch_dirs = sorted(OUTPUT_DIR.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
        if epoch_dirs:
            latest = epoch_dirs[-1]
            start_epoch = int(latest.name.split("_")[1]) + 1
            log(f"Resuming from checkpoint: {latest} (starting epoch {start_epoch})")
            lora_ckpt = latest / "lora_weights.pt"
            if lora_ckpt.exists():
                lora_state = torch.load(lora_ckpt, map_location=device)
                for name, m in model.model.named_modules():
                    if isinstance(m, LoRALinear):
                        if f"{name}.lora_A" in lora_state:
                            m.lora_A.weight.data = lora_state[f"{name}.lora_A"].to(device)
                            m.lora_B.weight.data = lora_state[f"{name}.lora_B"].to(device)
                log(f"  LoRA weights loaded from {lora_ckpt}")
            uc = latest / "user_converter.pkl"
            ic = latest / "item_converter.pkl"
            if uc.exists():
                model.user_embedding_converter.load_state_dict(torch.load(uc, map_location=device))
                log("  User converter loaded")
            if ic.exists():
                model.item_embedding_converter.load_state_dict(torch.load(ic, map_location=device))
                log("  Item converter loaded")
            global_step = (start_epoch - 1) * len(dataloader) // GRAD_ACCUM
        else:
            log("No checkpoint found — starting from scratch")

    # ── Training loop ─────────────────────────────────────────────────────────
    log(f"\nStarting training: {EPOCHS} epochs, {len(dataloader)} batches/epoch")
    log(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    if cli_args.resume:
        log(f"Resuming from epoch {start_epoch}")

    model.train()

    for epoch in range(start_epoch, EPOCHS + 1):
        log(f"\n{'='*50}")
        log(f"EPOCH {epoch}/{EPOCHS}")
        log(f"{'='*50}")

        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            try:
                loss = training_step(model, batch, device)
                loss = loss / GRAD_ACCUM
                loss.backward()
                epoch_loss += loss.item() * GRAD_ACCUM

                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        avg = epoch_loss / (step + 1)
                        log(f"  Step {global_step} | Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

                    # Periodic checkpoint
                    if global_step % SAVE_EVERY == 0:
                        ckpt_dir = OUTPUT_DIR / f"step_{global_step}"
                        ckpt_dir.mkdir(exist_ok=True)
                        save_lora_weights(model.model, ckpt_dir / "lora_weights.pt")
                        torch.save(model.user_embedding_converter.state_dict(),
                                   ckpt_dir / "user_converter.pkl")
                        torch.save(model.item_embedding_converter.state_dict(),
                                   ckpt_dir / "item_converter.pkl")
                        log(f"  Checkpoint saved → {ckpt_dir}")

            except Exception as e:
                log(f"  Step {step} failed: {e}")
                traceback.print_exc()
                optimizer.zero_grad()
                continue

        avg_loss = epoch_loss / len(dataloader)
        log(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")

        # Save epoch checkpoint
        epoch_dir = OUTPUT_DIR / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        save_lora_weights(model.model, epoch_dir / "lora_weights.pt")
        torch.save(model.user_embedding_converter.state_dict(),
                   epoch_dir / "user_converter.pkl")
        torch.save(model.item_embedding_converter.state_dict(),
                   epoch_dir / "item_converter.pkl")
        log(f"Epoch {epoch} checkpoint saved → {epoch_dir}")

    log("\n✅ Training complete!")
    log(f"Final weights saved in {OUTPUT_DIR}")
