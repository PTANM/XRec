import torch
from transformers import LogitsProcessor


class ItemConstrainedLogitsProcessor(LogitsProcessor):
    """
    Applies feature-constrained logit biasing at every decoding step.

    For each token in the LLM vocabulary:
      • If the token is in Vitem  → add +positive_bias  (grounding boost)
      • If the token is in the blacklist → add -negative_bias (filler suppression)
      • Otherwise                → no change

    Applied BEFORE softmax, so downstream greedy / beam search
    sees the manipulated distribution.
    """

    def __init__(
        self,
        verified_token_ids: set[int],   # Vitem for this item
        blacklist_token_ids: set[int],   # generic filler token IDs
        positive_bias: float = 15.0,     # scalar added to verified tokens
        negative_bias: float = 10.0,     # scalar subtracted from blacklist tokens
        device: str = "cuda",
    ):
        super().__init__()
        self.positive_bias = positive_bias
        self.negative_bias = negative_bias

        # Build bias tensors once; reuse across all decoding steps
        # (size is vocab_size, determined lazily on first call)
        self._verified = torch.tensor(
            sorted(verified_token_ids), dtype=torch.long, device=device
        )
        self._blacklist = torch.tensor(
            sorted(blacklist_token_ids), dtype=torch.long, device=device
        )
        self._device = device
        self._bias_cache: dict[int, torch.Tensor] = {}   # cache by vocab_size

    def _get_bias_tensor(self, vocab_size: int) -> torch.Tensor:
        if vocab_size not in self._bias_cache:
            bias = torch.zeros(vocab_size, device=self._device)
            # Apply positive bias to verified tokens (within vocab range)
            valid_verified = self._verified[self._verified < vocab_size]
            bias[valid_verified] += self.positive_bias
            # Apply negative bias to blacklist tokens
            valid_blacklist = self._blacklist[self._blacklist < vocab_size]
            bias[valid_blacklist] -= self.negative_bias
            self._bias_cache[vocab_size] = bias
        return self._bias_cache[vocab_size]

    def __call__(
        self,
        input_ids: torch.LongTensor,   # (batch, seq_len)
        scores: torch.FloatTensor,     # (batch, vocab_size) — RAW logits
    ) -> torch.FloatTensor:
        """Called by HuggingFace generate() before softmax at every step."""
        vocab_size = scores.shape[-1]
        bias = self._get_bias_tensor(vocab_size)          # (vocab_size,)
        scores = scores + bias.unsqueeze(0)               # broadcast over batch
        return scores