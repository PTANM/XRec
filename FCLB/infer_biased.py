import sys
sys.path.append('/scratch/user/kiarab/XRec/explainer')
sys.path.append('/scratch/user/kiarab/XRec')
sys.path.append('/scratch/user/kiarab/XRec/FCLB')

import pickle
import time
import torch
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args
from logit_bias import ItemConstrainedLogitsProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# -- Update these with your best tuning results --
POSITIVE_BIAS = 0.10
NEGATIVE_BIAS = 0.10
MIN_POSITION = 8

# -- Speed settings --
CHECKPOINT_INTERVAL = 500   
LOG_INTERVAL = 50           


class XRecBiased:
    def __init__(self):
        print(f"Dataset: {args.dataset}", flush=True)
        self.model = Explainer().to(device)
        self.data_handler = DataHandler()
        _, _, self.tst_loader = self.data_handler.load_data()

        base = f"/scratch/user/kiarab/XRec/FCLB/data/{args.dataset}"
        self.user_converter_path = f"{base}/user_converter.pkl"
        self.item_converter_path = f"{base}/item_converter.pkl"
        self.pred_path = f"{base}/tst_predictions_biased.pkl"
        self.ref_path = f"{base}/tst_references_biased.pkl"

        with open(f"{base}/item_vocab.pkl", "rb") as f:
            vocab_data = pickle.load(f)

        self.item_vocab = vocab_data["item_vocab"]
        self.blacklist_ids = vocab_data["blacklist_ids"]
        self.user_vocab = vocab_data.get("user_vocab", {})
        self.item_words = vocab_data.get("item_words", {})
        self.user_words = vocab_data.get("user_words", {})

        # SPEED: pre-compute all logit processors to avoid
        # redundant work inside the generation loop
        self._processor_cache = {}

    def _get_processor(self, uid, iid):
        """
        Cache processors by (uid, iid) pair. Many test samples
        share the same item, so this avoids recomputing the bias
        tensor hundreds of times.
        """
        cache_key = (uid, iid)
        if cache_key in self._processor_cache:
            return self._processor_cache[cache_key]

        # Compute preference-aligned token IDs
        item_ids = self.item_vocab.get(iid, set())
        pref_weight = 1.0

        if self.user_vocab:
            user_ids = self.user_vocab.get(uid, set())
            if user_ids:
                aligned = item_ids & user_ids
                if len(aligned) >= 5:
                    item_ids = aligned
                else:
                    pref_weight = 0.5

        # Compute word overlap weight
        u_words = self.user_words.get(uid, set())
        i_words = self.item_words.get(iid, set())
        if u_words and i_words:
            union = len(u_words | i_words)
            if union > 0:
                word_overlap = max(len(u_words & i_words) / union, 0.3)
            else:
                word_overlap = 1.0
        else:
            word_overlap = 1.0

        final_weight = (pref_weight + word_overlap) / 2.0

        processor = ItemConstrainedLogitsProcessor(
            verified_token_ids=item_ids,
            blacklist_token_ids=self.blacklist_ids,
            positive_bias=POSITIVE_BIAS,
            negative_bias=NEGATIVE_BIAS,
            preference_weight=final_weight,
            min_position=MIN_POSITION,
            device=str(device),
        )

        self._processor_cache[cache_key] = processor
        return processor

    def evaluate(self):
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_converter_path, map_location=device)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_converter_path, map_location=device)
        )
        self.model.eval()

        predictions = []
        references = []
        total = len(self.tst_loader)
        start_time = time.time()

        with torch.no_grad():
            for i, batch in enumerate(self.tst_loader):
                user_embed, item_embed, input_text, explain = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                iid = str(self.data_handler.tst_dict["iid"][i])
                uid = str(self.data_handler.tst_dict.get("uid", {}).get(i, ""))

                # SPEED: use cached processor
                logit_processor = self._get_processor(uid, iid)

                outputs = self.model.generate(
                    user_embed, item_embed, input_text,
                    logits_processor=[logit_processor],
                )

                end_idx = outputs[0].find("[")
                if end_idx != -1:
                    outputs[0] = outputs[0][:end_idx]

                predictions.append(outputs[0])
                references.append(explain[0])

                # SPEED: checkpoint less frequently
                if i > 0 and i % CHECKPOINT_INTERVAL == 0:
                    with open(self.pred_path + ".ckpt", "wb") as f:
                        pickle.dump(predictions, f)
                    with open(self.ref_path + ".ckpt", "wb") as f:
                        pickle.dump(references, f)

                # SPEED: log less frequently, but with ETA
                if i > 0 and i % LOG_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    per_sample = elapsed / i
                    remaining = per_sample * (total - i)
                    hrs = remaining / 3600
                    print(
                        f"Step [{i}/{total}] "
                        f"({per_sample:.1f}s/sample, "
                        f"ETA: {hrs:.1f}h)",
                        flush=True,
                    )

        with open(self.pred_path, "wb") as f:
            pickle.dump(predictions, f)
        with open(self.ref_path, "wb") as f:
            pickle.dump(references, f)

        total_time = (time.time() - start_time) / 3600
        print(f"Done. {len(predictions)} predictions in {total_time:.1f}h")
        print(f"Saved to {self.pred_path}", flush=True)


def main():
    runner = XRecBiased()
    print("Generating biased explanations (v3)...", flush=True)
    runner.evaluate()


if __name__ == "__main__":
    main()