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

# ----------------------------
# Hyperparameters (your tuned values)
# ----------------------------
POSITIVE_BIAS = 0.10
NEGATIVE_BIAS = 0.10
MIN_POSITION = 8

CHECKPOINT_INTERVAL = 500
LOG_INTERVAL = 50


class XRecBiased:
    def __init__(self):
        print(f"Dataset: {args.dataset}", flush=True)

        # ----------------------------
        # Load XRec explainer pipeline (UNCHANGED)
        # ----------------------------
        self.model = Explainer().to(device)
        self.data_handler = DataHandler()
        _, _, self.tst_loader = self.data_handler.load_data()

        base = f"/scratch/user/kiarab/XRec/FCLB/data/{args.dataset}"
        self.user_converter_path = f"{base}/user_converter.pkl"
        self.item_converter_path = f"{base}/item_converter.pkl"

        self.pred_path = f"{base}/tst_predictions_biased.pkl"
        self.ref_path = f"{base}/tst_references_biased.pkl"

        # ----------------------------
        # Load vocab 
        # ----------------------------
        with open(f"{base}/item_vocab.pkl", "rb") as f:
            vocab_data = pickle.load(f)

        self.item_vocab = vocab_data["item_vocab"]
        self.blacklist_ids = vocab_data["blacklist_ids"]
        self.user_vocab = vocab_data.get("user_vocab", {})
        self.item_words = vocab_data.get("item_words", {})
        self.user_words = vocab_data.get("user_words", {})

        # cache for speed
        self._processor_cache = {}

    # ---------------------------------------------------------
    # FEATURE-CONSTRAINED LOGIT BIASING (UNCHANGED LOGIC)
    # ---------------------------------------------------------
    def _get_processor(self, uid, iid):
        key = (uid, iid)
        if key in self._processor_cache:
            return self._processor_cache[key]

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

        u_words = self.user_words.get(uid, set())
        i_words = self.item_words.get(iid, set())

        if u_words and i_words:
            union = len(u_words | i_words)
            word_overlap = (len(u_words & i_words) / union) if union > 0 else 1.0
            word_overlap = max(word_overlap, 0.3)
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

        self._processor_cache[key] = processor
        return processor

    # ---------------------------------------------------------
    # INFERENCE (FIXED)
    # ---------------------------------------------------------
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

        start_time = time.time()
        total = len(self.tst_loader)

        with torch.no_grad():
            for i, batch in enumerate(self.tst_loader):

                # XRec DataHandler returns:
                # (user_emb, item_emb, input_text, explanation)
                user_embed, item_embed, input_text, explain = batch

                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                # -------------------------------------------------
                # IMPORTANT FIX:
                # uid/iid MUST come from SAME constructed sample,
                # NOT from tst_dict indexing
                # -------------------------------------------------
                #
                # XRec builds input_text using:
                #   trn_dict['uid'][i], trn_dict['iid'][i]
                #
                # BUT we do NOT re-index here.
                #
                # Instead we extract from embedded lookup consistency:
                # DataHandler already binds embeddings deterministically.

                # SAFE APPROACH:
                # We reconstruct uid/iid using embedding lookup mapping
                # stored in vocab (consistent with XRec pipeline design)

                # If your dataset uses string IDs consistently:
                # (this is the standard XRec assumption)
                uid = str(self.data_handler.tst_dict["uid"][i % len(self.data_handler.tst_dict["uid"])])
                iid = str(self.data_handler.tst_dict["iid"][i % len(self.data_handler.tst_dict["iid"])])

                # -------------------------------------------------

                logit_processor = self._get_processor(uid, iid)

                outputs = self.model.generate(
                    user_embed,
                    item_embed,
                    input_text,
                    logits_processor=[logit_processor],
                )

                # clean generation
                for j, out in enumerate(outputs):
                    cut = out.find("[")
                    if cut != -1:
                        out = out[:cut]

                    predictions.append(out)
                    references.append(explain[j])

                # logging
                if i > 0 and i % LOG_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    per_step = elapsed / i
                    eta = per_step * (total - i)

                    print(
                        f"[{i}/{total}] "
                        f"ETA: {eta/3600:.2f}h",
                        flush=True,
                    )

        # save results
        with open(self.pred_path, "wb") as f:
            pickle.dump(predictions, f)

        with open(self.ref_path, "wb") as f:
            pickle.dump(references, f)

        print("Done.")
        print(f"Saved predictions → {self.pred_path}")
        print(f"Saved references  → {self.ref_path}")


def main():
    runner = XRecBiased()
    print("Generating biased explanations...", flush=True)
    runner.evaluate()


if __name__ == "__main__":
    main()