"""
Drop-in replacement for the original XRec explainer.py.
Changes vs. original:
  1. MoE adapter  →  nn.Sequential 2-layer MLP (SimplifiedAdapter)
  2. NLL loss objective
  3. generate() uses ItemConstrainedLogitsProcessor for logit biasing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType

from explainer.logit_bias import ItemConstrainedLogitsProcessor


# ── Simplified Adapter (replaces MoE) ───────────────────────────────────────
class SimplifiedAdapter(nn.Module):
    """
    2-layer MLP that projects LightGCN collaborative embeddings
    into the LLM's token embedding space.

    LightGCN emb dim  →  hidden_dim  →  llm_hidden_size
    """

    def __init__(
        self,
        collab_emb_dim: int,          # LightGCN output dimension
        llm_hidden_size: int,         # LLM's hidden state dimension
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(collab_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_hidden_size),
        )

    def forward(self, collab_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            collab_embs: (batch, collab_emb_dim)
        Returns:
            projected: (batch, 1, llm_hidden_size)  — one soft token per item/user
        """
        projected = self.mlp(collab_embs)          # (batch, llm_hidden_size)
        return projected.unsqueeze(1)              # (batch, 1, llm_hidden_size)


# ── Main Model ───────────────────────────────────────────────────────────────
class XRecMLPExplainer(nn.Module):
    def __init__(
        self,
        llm_model_name: str,
        collab_emb_dim: int,
        adapter_hidden_dim: int = 512,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # ── Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ── LLM backbone (LLaMA-2) with LoRA ─────────────────────────────
        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        self.llm = get_peft_model(base_llm, lora_config)
        llm_hidden_size = base_llm.config.hidden_size

        # ── Simplified MLP Adapter ─────────────────────────────────────────
        self.adapter = SimplifiedAdapter(
            collab_emb_dim=collab_emb_dim,
            llm_hidden_size=llm_hidden_size,
            hidden_dim=adapter_hidden_dim,
        ).to(device)

        # ── Token embedding layer (for text portion of input) ──────────────
        self.token_emb = self.llm.get_input_embeddings()

    # ─────────────────────────────────────────────────────────────────────────
    def _build_inputs_embeds(
        self,
        user_collab_emb: torch.Tensor,   # (batch, collab_emb_dim)
        item_collab_emb: torch.Tensor,   # (batch, collab_emb_dim)
        prompt_input_ids: torch.LongTensor,  # (batch, prompt_seq_len)
    ) -> torch.Tensor:
        """
        Concatenate [user_soft_token | item_soft_token | prompt_token_embs]
        along the sequence dimension.
        """
        user_soft = self.adapter(user_collab_emb)          # (batch, 1, hidden)
        item_soft = self.adapter(item_collab_emb)          # (batch, 1, hidden)
        text_embs = self.token_emb(prompt_input_ids)       # (batch, seq, hidden)
        inputs_embeds = torch.cat(
            [user_soft, item_soft, text_embs], dim=1
        )                                                   # (batch, 2+seq, hidden)
        return inputs_embeds

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        user_collab_emb: torch.Tensor,
        item_collab_emb: torch.Tensor,
        prompt_input_ids: torch.LongTensor,
        prompt_attention_mask: torch.LongTensor,
        label_input_ids: torch.LongTensor,      # (batch, label_seq_len)
    ) -> torch.Tensor:
        """
        Training forward pass. Returns NLL loss.

        Labels use -100 for prompt positions (masked from loss).
        """
        inputs_embeds = self._build_inputs_embeds(
            user_collab_emb, item_collab_emb, prompt_input_ids
        )

        # Extend attention mask to cover the 2 soft tokens prepended
        batch_size = prompt_input_ids.size(0)
        soft_mask = torch.ones(
            batch_size, 2, dtype=torch.long, device=self.device
        )
        full_attention_mask = torch.cat(
            [soft_mask, prompt_attention_mask], dim=1
        )  # (batch, 2+prompt_seq)

        # Shift labels: align with the full input sequence
        # Mask soft-token positions and prompt positions with -100
        soft_label_mask = torch.full(
            (batch_size, 2 + prompt_input_ids.size(1)),
            fill_value=-100,
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.cat([soft_label_mask, label_input_ids], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,       # HuggingFace computes NLL internally when labels are provided
            return_dict=True,
        )

        # outputs.loss is already the mean NLL (cross-entropy) over non-masked positions
        return outputs.loss

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def generate_explanation(
        self,
        user_collab_emb: torch.Tensor,          # (1, collab_emb_dim)
        item_collab_emb: torch.Tensor,          # (1, collab_emb_dim)
        prompt_input_ids: torch.LongTensor,     # (1, prompt_seq_len)
        prompt_attention_mask: torch.LongTensor,
        verified_token_ids: set[int],           # Vitem for this item
        blacklist_token_ids: set[int],          # shared blacklist
        positive_bias: float = 15.0,
        negative_bias: float = 10.0,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Feature-Constrained greedy decoding.

        1. Build inputs_embeds (soft tokens + prompt)
        2. Attach ItemConstrainedLogitsProcessor
        3. Greedy decode → decoded string
        """
        inputs_embeds = self._build_inputs_embeds(
            user_collab_emb, item_collab_emb, prompt_input_ids
        )

        batch_size = prompt_input_ids.size(0)
        soft_mask  = torch.ones(batch_size, 2, dtype=torch.long, device=self.device)
        full_mask  = torch.cat([soft_mask, prompt_attention_mask], dim=1)

        # ── Instantiate the logit biasing processor ────────────────────────
        logit_processor = ItemConstrainedLogitsProcessor(
            verified_token_ids=verified_token_ids,
            blacklist_token_ids=blacklist_token_ids,
            positive_bias=positive_bias,
            negative_bias=negative_bias,
            device=self.device,
        )

        generation_config = GenerationConfig(
            do_sample=False,          # greedy decoding
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            generation_config=generation_config,
            logits_processor=[logit_processor],   # ← injected here
        )

        # Decode only newly generated tokens (skip prompt echo)
        prompt_len = inputs_embeds.size(1)
        new_tokens  = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)