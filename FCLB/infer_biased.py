import sys
sys.path.append('/scratch/user/kiarab/XRec/explainer')
sys.path.append('/scratch/user/kiarab/XRec')

import pickle
import torch
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args
from logit_bias import ItemConstrainedLogitsProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

POSITIVE_BIAS = 0.15
NEGATIVE_BIAS = 0.0
MIN_POSITION = 8


class XRecBiased:
    def __init__(self):
        print(f"Dataset: {args.dataset}")
        self.model = Explainer().to(device)
        self.data_handler = DataHandler()
        _, _, self.tst_loader = self.data_handler.load_data()

        base = f"/scratch/user/kiarab/XRec/data/{args.dataset}"
        self.user_converter_path = f"{base}/user_converter.pkl"
        self.item_converter_path = f"{base}/item_converter.pkl"
        self.pred_path = f"{base}/tst_predictions_biased.pkl"
        self.ref_path = f"{base}/tst_references_biased.pkl"

        with open(f"{base}/item_vocab.pkl", "rb") as f:
            vocab_data = pickle.load(f)

        self.item_vocab = vocab_data["item_vocab"]
        self.blacklist_ids = vocab_data["blacklist_ids"]

        # New: user vocab for preference-aligned intersection
        self.user_vocab = vocab_data.get("user_vocab", {})
        self.item_words = vocab_data.get("item_words", {})
        self.user_words = vocab_data.get("user_words", {})

    def _get_preference_aligned_ids(self, uid, iid):
        """
        If we have user vocab, return intersection of user and item
        token IDs (preference-aligned tokens). Otherwise fall back
        to item-only vocab.
        """
        item_ids = self.item_vocab.get(iid, set())
        if not self.user_vocab:
            return item_ids, 1.0

        user_ids = self.user_vocab.get(uid, set())
        if not user_ids:
            return item_ids, 1.0

        aligned = item_ids & user_ids
        if len(aligned) < 5:
            return item_ids, 0.5
        return aligned, 1.0

    def _compute_word_overlap(self, uid, iid):
        """
        Compute word-level overlap ratio between user and item
        preference words. Used as preference_weight.
        """
        u_words = self.user_words.get(uid, set())
        i_words = self.item_words.get(iid, set())
        if not u_words or not i_words:
            return 1.0
        overlap = len(u_words & i_words)
        union = len(u_words | i_words)
        if union == 0:
            return 1.0
        return max(overlap / union, 0.3)

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

        with torch.no_grad():
            for i, batch in enumerate(self.tst_loader):
                user_embed, item_embed, input_text, explain = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                iid = str(self.data_handler.tst_dict["iid"][i])
                uid = str(self.data_handler.tst_dict.get("uid", {}).get(i, ""))

                verified_ids, pref_weight_from_ids = (
                    self._get_preference_aligned_ids(uid, iid)
                )
                word_overlap = self._compute_word_overlap(uid, iid)
                final_weight = (pref_weight_from_ids + word_overlap) / 2.0

                logit_processor = ItemConstrainedLogitsProcessor(
                    verified_token_ids=verified_ids,
                    blacklist_token_ids=self.blacklist_ids,
                    positive_bias=POSITIVE_BIAS,
                    negative_bias=NEGATIVE_BIAS,
                    preference_weight=final_weight,
                    min_position=MIN_POSITION,
                    device=str(device),
                )

                outputs = self.model.generate(
                    user_embed, item_embed, input_text,
                    logits_processor=[logit_processor],
                )

                end_idx = outputs[0].find("[")
                if end_idx != -1:
                    outputs[0] = outputs[0][:end_idx]

                predictions.append(outputs[0])
                references.append(explain[0])

                if i % 100 == 0:
                    with open(self.pred_path + ".ckpt", "wb") as f:
                        pickle.dump(predictions, f)
                    with open(self.ref_path + ".ckpt", "wb") as f:
                        pickle.dump(references, f)
                    print(f"Checkpoint at step {i}/{len(self.tst_loader)}", flush=True)

                if i % 10 == 0 and i > 0:
                    print(f"Step [{i}/{len(self.tst_loader)}]", flush=True)
                    print(f"  Output: {outputs[0][:120]}...", flush=True)

        with open(self.pred_path, "wb") as f:
            pickle.dump(predictions, f)
        with open(self.ref_path, "wb") as f:
            pickle.dump(references, f)
        print(f"Saved {len(predictions)} predictions to {self.pred_path}")


def main():
    runner = XRecBiased()
    print("Generating biased explanations (v2)...")
    runner.evaluate()


if __name__ == "__main__":
    main()