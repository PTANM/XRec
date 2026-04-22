import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from transformers import LlamaTokenizer

from models.modeling_explainer import LlamaForCausalLM
from utils.parse import args


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer."""

    def __init__(self, input_size, output_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """Original XRec MoE-enhanced adaptor."""

    def __init__(self, n_exps=8, layers=None, dropout=0.2, noise=True):
        super().__init__()
        if layers is None:
            layers = [64, 4096]
        self.n_exps = n_exps
        self.noisy_gating = noise
        self.experts = nn.ModuleList(
            [PWLayer(layers[0], layers[1], dropout) for _ in range(n_exps)]
        )
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits
        return F.softmax(logits, dim=-1)

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)
        expert_outputs = [expert(x).unsqueeze(-2) for expert in self.experts]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class MLPAdaptorLayer(nn.Module):
    """Standard 2-layer MLP adaptor used in the ablation."""

    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.network(x)


class Explainer(nn.Module):
    def __init__(self, token_size=4096, user_embed_size=64, item_embed_size=64):
        super().__init__()

        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if args.load_in_8bit and torch.cuda.is_available():
            model_kwargs["load_in_8bit"] = True

        self.model = LlamaForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        self.tokenizer = LlamaTokenizer.from_pretrained(args.model_name)

        special_tokens_dict = {
            "additional_special_tokens": ["<USER_EMBED>", "<ITEM_EMBED>", "<EXPLAIN_POS>"]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token = "<pad>"
        self.model.resize_token_embeddings(len(self.tokenizer))

        for param in self.model.parameters():
            param.requires_grad = False

        self.adapter_device = self.model.get_input_embeddings().weight.device
        self.adapter_dtype = self.model.get_input_embeddings().weight.dtype

        if torch.cuda.is_available():
            visible_gpus = ", ".join(
                [
                    f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
                    for idx in range(torch.cuda.device_count())
                ]
            )
            print(f"CUDA available. Visible GPUs: {visible_gpus}")
        else:
            print("CUDA unavailable. Falling back to CPU.")
        print(f"Model input embedding device: {self.adapter_device}")
        print(f"Model input embedding dtype: {self.adapter_dtype}")

        self.user_embedding_converter = self._build_adapter(
            user_embed_size, token_size
        ).to(self.adapter_device)
        self.item_embedding_converter = self._build_adapter(
            item_embed_size, token_size
        ).to(self.adapter_device)

    def _build_adapter(self, input_size, output_size):
        if args.adapter_type == "mlp":
            return MLPAdaptorLayer(
                input_size=input_size,
                hidden_size=args.adapter_hidden_size,
                output_size=output_size,
                dropout=args.adapter_dropout,
            )
        return MoEAdaptorLayer(
            n_exps=8,
            layers=[input_size, output_size],
            dropout=args.adapter_dropout,
            noise=True,
        )

    def adapter_parameters(self):
        return list(self.user_embedding_converter.parameters()) + list(
            self.item_embedding_converter.parameters()
        )

    def _prepare_inputs(self, user_embedding, item_embedding, input_text):
        user_embedding = user_embedding.to(
            device=self.adapter_device, dtype=self.adapter_dtype
        )
        item_embedding = item_embedding.to(
            device=self.adapter_device, dtype=self.adapter_dtype
        )
        converted_user_embedding = self.user_embedding_converter(user_embedding).to(
            dtype=self.adapter_dtype
        )
        converted_item_embedding = self.item_embedding_converter(item_embedding).to(
            dtype=self.adapter_dtype
        )

        tokenized_inputs = self.tokenizer(
            input_text, padding=True, return_tensors="pt"
        )
        input_ids = tokenized_inputs["input_ids"].to(self.adapter_device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.adapter_device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        user_embed_token_id = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        item_embed_token_id = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        explain_pos_token_id = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")

        user_embed_position = (input_ids == user_embed_token_id).nonzero()[:, 1:]
        item_embed_position = (input_ids == item_embed_token_id).nonzero()[:, 1:]
        explain_pos_position = (input_ids == explain_pos_token_id).nonzero()[:, 1:]

        batch_indices = torch.arange(user_embed_position.shape[0], device=self.adapter_device)
        inputs_embeds[batch_indices, user_embed_position[:, 0], :] = converted_user_embedding
        inputs_embeds[batch_indices, item_embed_position[:, 0], :] = converted_item_embedding

        return (
            input_ids,
            attention_mask,
            inputs_embeds,
            converted_user_embedding,
            converted_item_embedding,
            user_embed_position,
            item_embed_position,
            explain_pos_position.flatten(),
        )

    def forward(self, user_embedding, item_embedding, input_text):
        (
            input_ids,
            attention_mask,
            inputs_embeds,
            converted_user_embedding,
            converted_item_embedding,
            user_embed_position,
            item_embed_position,
            explain_pos_position,
        ) = self._prepare_inputs(user_embedding, item_embedding, input_text)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            user_embed=converted_user_embedding,
            item_embed=converted_item_embedding,
            user_embed_pos=user_embed_position,
            item_embed_pos=item_embed_position,
        )
        return input_ids, outputs, explain_pos_position

    def loss(self, input_ids, outputs, explain_pos_position):
        labels = input_ids.clone()
        interval = torch.arange(labels.shape[1], device=labels.device)
        mask = interval[None, :] < explain_pos_position[:, None]
        labels[mask] = -100

        logits = outputs.logits.float()
        shift_labels = labels[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        loss = nn.NLLLoss(ignore_index=-100)(
            log_probs.view(-1, log_probs.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    def generate(self, user_embedding, item_embedding, input_text):
        (
            _,
            attention_mask,
            inputs_embeds,
            converted_user_embedding,
            converted_item_embedding,
            user_embed_position,
            item_embed_position,
            _,
        ) = self._prepare_inputs(user_embedding, item_embedding, input_text)

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            user_embed=converted_user_embedding,
            item_embed=converted_item_embedding,
            user_embed_pos=user_embed_position,
            item_embed_pos=item_embed_position,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
