import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from explainer.models.explainer_mlp import XRecMLPExplainer
from explainer.utils.dataset import XRecDataset   # existing XRec dataset util


def train(config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load pre-built vocab ───────────────────────────────────────────────
    with open(config["vocab_path"], "rb") as f:
        vocab_data = pickle.load(f)
    blacklist_ids = vocab_data["blacklist_ids"]
    # item_vocab used at inference time, not training

    # ── Model ──────────────────────────────────────────────────────────────
    model = XRecMLPExplainer(
        llm_model_name=config["llm_model_name"],
        collab_emb_dim=config["collab_emb_dim"],
        adapter_hidden_dim=config.get("adapter_hidden_dim", 512),
        device=device,
    )

    # Load pre-trained LightGCN embeddings (frozen)
    with open(config["user_emb_path"], "rb") as f:
        user_embs = torch.tensor(pickle.load(f), dtype=torch.float32).to(device)
    with open(config["item_emb_path"], "rb") as f:
        item_embs = torch.tensor(pickle.load(f), dtype=torch.float32).to(device)

    # Only the adapter + LoRA weights are trainable
    trainable_params = (
        list(model.adapter.parameters()) +
        [p for n, p in model.llm.named_parameters() if p.requires_grad]
    )
    optimizer = AdamW(trainable_params, lr=config.get("lr", 2e-4), weight_decay=1e-2)

    dataset    = XRecDataset(config["train_data_path"], model.tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=config.get("batch_size", 8), shuffle=True, num_workers=4
    )

    total_steps = len(dataloader) * config.get("epochs", 3)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(config.get("epochs", 3)):
        for step, batch in enumerate(dataloader):
            user_ids  = batch["user_id"].to(device)
            item_ids  = batch["item_id"].to(device)

            user_collab = user_embs[user_ids]    # (batch, collab_emb_dim)
            item_collab = item_embs[item_ids]    # (batch, collab_emb_dim)

            loss = model(
                user_collab_emb=user_collab,
                item_collab_emb=item_collab,
                prompt_input_ids=batch["prompt_input_ids"].to(device),
                prompt_attention_mask=batch["prompt_attention_mask"].to(device),
                label_input_ids=batch["label_input_ids"].to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | NLL Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), config["output_model_path"])
    print("Training complete.")