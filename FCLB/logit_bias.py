import torch
from transformers import LogitsProcessor


class ItemConstrainedLogitsProcessor(LogitsProcessor):
    """
    Logits processor that steers generation toward item/user preference vocabulary.

    Key improvements over v1:
      1. Entropy gating  — bias only fires when the model is uncertain (high
         entropy). When the model is confident (mid-word, grammatical slot),
         we leave the distribution alone to preserve fluency.
      2. Decaying bias   — the bias starts at full strength and tapers off
         as the sequence grows, so topic words are seeded early without
         distorting later refinements.
      3. Both mechanisms are multiplicative, so they compose cleanly.
    """

    def __init__(
        self,
        verified_token_ids: set,
        blacklist_token_ids: set = None,
        positive_bias: float = 0.15,
        negative_bias: float = 0.0,
        preference_weight: float = 1.0,
        min_position: int = 8,
        decay_horizon: int = 60,   # steps over which bias decays to `decay_floor`
        decay_floor: float = 0.2,  # minimum bias multiplier (never fully zeroed)
        entropy_threshold: float = 0.3,  # fraction of max-entropy below which bias is suppressed
        device: str = "cuda",
    ):
        super().__init__()
        self.positive_bias = positive_bias * preference_weight
        self.negative_bias = negative_bias * preference_weight
        self.min_position = min_position
        self.decay_horizon = decay_horizon
        self.decay_floor = decay_floor
        self.entropy_threshold = entropy_threshold

        if verified_token_ids:
            self._verified = torch.tensor(
                sorted(verified_token_ids), dtype=torch.long, device=device
            )
        else:
            self._verified = torch.tensor([], dtype=torch.long, device=device)

        if blacklist_token_ids:
            self._blacklist = torch.tensor(
                sorted(blacklist_token_ids), dtype=torch.long, device=device
            )
        else:
            self._blacklist = torch.tensor([], dtype=torch.long, device=device)

        self._device = device
        self._bias_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_bias_tensor(self, vocab_size: int) -> torch.Tensor:
        """Build (and cache) the base bias vector for a given vocab size."""
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

    def _decay_multiplier(self, seq_len: int) -> float:
        """
        Linear decay from 1.0 → decay_floor over `decay_horizon` steps,
        then held constant at decay_floor.

        Example with defaults (horizon=60, floor=0.2):
          position 8  → 1.00  (full bias)
          position 38 → 0.60
          position 68 → 0.20  (floor, stays here)
        """
        steps_past_start = max(0, seq_len - self.min_position)
        frac = min(steps_past_start / self.decay_horizon, 1.0)
        return 1.0 - frac * (1.0 - self.decay_floor)

    def _entropy_gate(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns a per-batch gate in [0, 1].

        Gate = 0  → model is very confident (e.g. mid-word); suppress bias.
        Gate = 1  → model is maximally uncertain; apply full bias.

        Specifically: gate = max(0, (H - threshold * H_max) / ((1-threshold) * H_max))
        This is 0 below the threshold and ramps linearly to 1 at H_max.
        """
        probs = torch.softmax(scores.float(), dim=-1)
        # Shannon entropy H in nats; clamp for numerical safety
        H = -(probs * (probs + 1e-10).log()).sum(dim=-1, keepdim=True)   # (B, 1)
        H_max = torch.log(torch.tensor(scores.shape[-1], dtype=torch.float,
                                       device=self._device))
        # Normalised entropy in [0, 1]
        H_norm = (H / H_max).clamp(0.0, 1.0)
        # Linear ramp above threshold → gate in [0, 1]
        gate = ((H_norm - self.entropy_threshold) /
                (1.0 - self.entropy_threshold + 1e-8)).clamp(0.0, 1.0)
        return gate  # shape (B, 1)

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        seq_len = input_ids.shape[-1]
        if seq_len < self.min_position:
            return scores

        vocab_size = scores.shape[-1]
        base_bias = self._get_bias_tensor(vocab_size)          # (V,)

        # 1. Temporal decay — stronger early, weaker late
        decay = self._decay_multiplier(seq_len)                # scalar

        # 2. Entropy gate — suppressed when model is confident
        gate = self._entropy_gate(scores)                      # (B, 1)

        # Combined scale: (B, 1) broadcast over vocab
        scale = decay * gate                                   # (B, 1)

        return scores + base_bias.unsqueeze(0) * scale
