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

POSITIVE_BIAS = 0.10
NEGATIVE_BIAS = 0.10
MIN_POSITION = 8

CHECKPOINT_INTERVAL = 500
LOG_INTERVAL = 50


class XRecBiased:
    def __init__(self):

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

        self._processor_cache = {}

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
            word_overlap = max(len(u_words & i_words) / union, 0.3) if union else 1.0
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

    def evaluate(self):

        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_converter_path, map_location=device)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_converter_path, map_location=device)
        )
        self.model.eval()

        predictions, references = [], []
        start = time.time()
        total = len(self.tst_loader)

        model_generate = self.model.generate  # 🔥 speed

        with torch.inference_mode():

            for i, batch in enumerate(self.tst_loader):

                user_embed, item_embed, input_text, explain = batch

                user_embed = user_embed.to(device, non_blocking=True).contiguous()
                item_embed = item_embed.to(device, non_blocking=True).contiguous()

                iid = str(self.data_handler.tst_dict["iid"][i])
                uid = str(self.data_handler.tst_dict["uid"][i])

                logits_processor = [self._get_processor(uid, iid)]

                outputs = model_generate(
                    user_embed,
                    item_embed,
                    input_text,
                    logits_processor=logits_processor,
                    use_cache=True,
                    num_beams=1,
                )

                text = outputs[0]
                cut = text.find("[")
                if cut != -1:
                    text = text[:cut]

                predictions.append(text)
                references.append(explain[0])

                if i % LOG_INTERVAL == 0 and i > 0:
                    elapsed = time.time() - start
                    print(f"[{i}/{total}] ETA: {(elapsed/i)*(total-i)/3600:.2f}h")

                if i % CHECKPOINT_INTERVAL == 0 and i > 0:
                    with open(self.pred_path + ".ckpt", "wb") as f:
                        pickle.dump(predictions, f)
                    with open(self.ref_path + ".ckpt", "wb") as f:
                        pickle.dump(references, f)

        with open(self.pred_path, "wb") as f:
            pickle.dump(predictions, f)

        with open(self.ref_path, "wb") as f:
            pickle.dump(references, f)

        print(f"Done in {(time.time()-start)/3600:.2f}h")