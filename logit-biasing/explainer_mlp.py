"""
Drop-in replacement for the original XRec explainer.py.
Changes vs. original:
  1. MoE adapter  →  nn.Sequential 2-layer MLP (SimplifiedAdapter)
  2. NLL loss objective
  3. generate() uses ItemConstrainedLogitsProcessor for logit biasing
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import get_peft_model, LoraConfig, TaskType

from logit_bias import ItemConstrainedLogitsProcessor


class SimplifiedAdapter(nn.Module):
    def __init__(self, collab_emb_dim: int, llm_hidden_size: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(collab_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_hidden_size),
        )

    def forward(self, collab_embs: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(collab_embs)
        return projected.unsqueeze(1)


class XRecMLPExplainer(nn.Module):
    def __init__(self, llm_model_name: str, collab_emb_dim: int, adapter_hidden_dim: int = 512,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
                 device: str = "cuda", lora_target_modules: list = ["q_proj", "v_proj"]):
        super().__init__()
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=lora_target_modules,
        )
        self.llm = get_peft_model(base_llm, lora_config)
        llm_hidden_size = base_llm.config.hidden_size

        self.adapter = SimplifiedAdapter(
            collab_emb_dim=collab_emb_dim, llm_hidden_size=llm_hidden_size, hidden_dim=adapter_hidden_dim
        ).to(device)

        self.token_emb = self.llm.get_input_embeddings()

    def _build_inputs_embeds(self, user_collab_emb, item_collab_emb, prompt_input_ids):
        user_soft  = self.adapter(user_collab_emb).float()
        item_soft  = self.adapter(item_collab_emb).float()
        text_embs  = self.token_emb(prompt_input_ids).float()
        return torch.cat([user_soft, item_soft, text_embs], dim=1)

    def forward(self, user_collab_emb, item_collab_emb, prompt_input_ids,
                prompt_attention_mask, label_input_ids):
        inputs_embeds = self._build_inputs_embeds(user_collab_emb, item_collab_emb, prompt_input_ids)
        batch_size = prompt_input_ids.size(0)

        soft_mask = torch.ones(batch_size, 2, dtype=torch.long, device=self.device)
        full_attention_mask = torch.cat([soft_mask, prompt_attention_mask], dim=1)

        soft_label_mask = torch.full(
            (batch_size, 2 + prompt_input_ids.size(1)), fill_value=-100,
            dtype=torch.long, device=self.device
        )
        labels = torch.cat([soft_label_mask, label_input_ids], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=full_attention_mask,
            labels=labels, return_dict=True
        )
        return outputs.loss

    @torch.inference_mode()
    def generate_explanation(self, user_collab_emb, item_collab_emb, prompt_input_ids,
                              prompt_attention_mask, verified_token_ids, blacklist_token_ids,
                              positive_bias=15.0, negative_bias=10.0, max_new_tokens=128):
        inputs_embeds = self._build_inputs_embeds(user_collab_emb, item_collab_emb, prompt_input_ids)
        batch_size = prompt_input_ids.size(0)
        soft_mask  = torch.ones(batch_size, 2, dtype=torch.long, device=self.device)
        full_mask  = torch.cat([soft_mask, prompt_attention_mask], dim=1)

        logit_processor = ItemConstrainedLogitsProcessor(
            verified_token_ids=verified_token_ids, blacklist_token_ids=blacklist_token_ids,
            positive_bias=positive_bias, negative_bias=negative_bias, device=self.device,
        )
        generation_config = GenerationConfig(
            do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=full_mask,
            generation_config=generation_config, logits_processor=[logit_processor],
        )
        prompt_len = inputs_embeds.size(1)
        new_tokens = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)