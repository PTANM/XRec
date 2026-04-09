# train_mlp.py

import sys
sys.path.append('../explainer')
sys.path.append('../')

import pickle
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from explainer_mlp import XRecMLPExplainer
from utils.data_handler import DataHandler


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = XRecMLPExplainer(
        llm_model_name="meta-llama/Llama-2-7b-hf",
        collab_emb_dim=64,
        device=device,
    )

    # Use original DataHandler (reads args.dataset, args.batch_size etc.)
    data_handler = DataHandler()
    trn_loader, val_loader, tst_loader = data_handler.load_data()

    # Only train adapter + LoRA weights
    trainable_params = (
        list(model.adapter.parameters()) +
        [p for n, p in model.llm.named_parameters() if p.requires_grad]
    )
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-6)

    total_steps = len(trn_loader) * 1  # epochs handled by args
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    for epoch in range(1):  # use args.epochs if you want
        for step, batch in enumerate(trn_loader):
            user_emb, item_emb, prompt = batch

            user_collab = user_emb.to(device).float()
            item_collab = item_emb.to(device).float()

            # Tokenize prompt
            encoded = model.tokenizer(
                list(prompt), return_tensors="pt", padding=True,
                truncation=True, max_length=512
            )

            # Extract label portion (after <EXPLAIN_POS>)
            label_text = [p.split("<EXPLAIN_POS>")[-1].strip() for p in prompt]
            label_enc  = model.tokenizer(
                label_text, return_tensors="pt", padding=True,
                truncation=True, max_length=128
            )

            loss = model(
                user_collab_emb=user_collab,
                item_collab_emb=item_collab,
                prompt_input_ids=encoded["input_ids"].to(device),
                prompt_attention_mask=encoded["attention_mask"].to(device),
                label_input_ids=label_enc["input_ids"].to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | NLL Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "../checkpoints/xrec_mlp.pt")
    print("Training complete.")


if __name__ == "__main__":
    train()