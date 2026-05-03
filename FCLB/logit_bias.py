import torch
from transformers import LogitsProcessor

# Logit processor class to be added to 
class ItemConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        verified_token_ids: set,
        blacklist_token_ids: set = None,
        positive_bias: float = 0.15,
        negative_bias: float = 0.0,
        preference_weight: float = 1.0,
        min_position: int = 8,
        device: str = "cuda",
    ):
        super().__init__()
        self.positive_bias = positive_bias * preference_weight
        self.negative_bias = negative_bias * preference_weight
        self.min_position = min_position

        if verified_token_ids:
            self._verified = torch.tensor(sorted(verified_token_ids), dtype=torch.long, device=device)
        else:
            self._verified = torch.tensor([], dtype=torch.long, device=device)

        if blacklist_token_ids:
            self._blacklist = torch.tensor(sorted(blacklist_token_ids), dtype=torch.long, device=device)
        else:
            self._blacklist = torch.tensor([], dtype=torch.long, device=device)

        self._device = device
        self._bias_cache = {}

    def _get_bias_tensor(self, vocab_size: int) -> torch.Tensor:
        if vocab_size not in self._bias_cache:
            bias = torch.zeros(vocab_size, device=self._device)
            if len(self._verified) > 0:
                valid = self._verified[self._verified < vocab_size]
                bias[valid] += self.positive_bias
            if self.negative_bias > 0 and len(self._blacklist) > 0:
                valid_bl = self._blacklist[self._blacklist < vocab_size]
                bias[valid_bl] -= self.negative_bias
            self._bias_cache[vocab_size] = bias
        return self._bias_cache[vocab_size]

    def __call__(self,input_ids: torch.LongTensor,scores: torch.FloatTensor,) -> torch.FloatTensor:
        seq_len = input_ids.shape[-1]
        if seq_len < self.min_position:
            return scores
        vocab_size = scores.shape[-1]
        bias = self._get_bias_tensor(vocab_size)
        return scores + bias.unsqueeze(0)


class SafeItemLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        verified_token_ids,
        blacklist_token_ids,
        positive_bias=0.02,
        negative_bias=0.01,
        min_position=15,
        top_k=50,
    ):
        self.verified_token_ids = list(verified_token_ids)
        self.blacklist_token_ids = list(blacklist_token_ids)

        self.pos_bias = positive_bias
        self.neg_bias = negative_bias

        self.min_position = min_position
        self.top_k = top_k

    def __call__(self, input_ids, scores):
        """
        scores: (batch_size, vocab_size)
        """

        step = input_ids.shape[1]

        # Skip few steps at beginning
        if step < self.min_position:
            return scores

        # Prevent numerical explosion
        scores = torch.clamp(scores, -50, 50)

        # Loop per batch
        for b in range(scores.size(0)):

            logits = scores[b]

            # Top-k safety filter
            topk_vals, topk_idx = torch.topk(logits, self.top_k)

            # Create mask for allowed tokens
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[topk_idx] = True

            # Apply POSITIVE bias
            if self.verified_token_ids:
                valid_ids = torch.tensor(
                    self.verified_token_ids,
                    device=logits.device
                )

                valid_ids = valid_ids[valid_ids < logits.size(0)]

                valid_ids = valid_ids[mask[valid_ids]]

                logits[valid_ids] += self.pos_bias

            # Apply NEGATIVE bias 
            if self.blacklist_token_ids:
                bad_ids = torch.tensor(
                    self.blacklist_token_ids,
                    device=logits.device
                )

                bad_ids = bad_ids[bad_ids < logits.size(0)]
                bad_ids = bad_ids[mask[bad_ids]]

                logits[bad_ids] -= self.neg_bias

            scores[b] = logits

        return scores