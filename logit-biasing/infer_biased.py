# infer_biased.py

import sys
sys.path.append('../explainer')
sys.path.append('../')

import pickle
import torch
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args
from logit_bias import ItemConstrainedLogitsProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")


class XRecBiased:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        self.model = Explainer().to(device)
        self.data_handler = DataHandler()

        _, _, self.tst_loader = self.data_handler.load_data()

        self.user_embedding_converter_path = f"../data/{args.dataset}/user_converter.pkl"
        self.item_embedding_converter_path = f"../data/{args.dataset}/item_converter.pkl"
        self.tst_predictions_path = f"../data/{args.dataset}/tst_predictions_biased.pkl"
        self.tst_references_path  = f"../data/{args.dataset}/tst_references_biased.pkl"

        # Load item vocab for logit biasing
        with open(f"../data/{args.dataset}/item_vocab.pkl", "rb") as f:
            vocab_data = pickle.load(f)
        self.item_vocab    = vocab_data["item_vocab"]
        self.blacklist_ids = vocab_data["blacklist_ids"]

    def evaluate(self):
        # Load pre-trained MoE converter weights
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path, map_location=device)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path, map_location=device)
        )
        self.model.eval()

        predictions = []
        references  = []

        with torch.no_grad():
            for i, batch in enumerate(self.tst_loader):
                user_embed, item_embed, input_text, explain = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                # Look up verified token IDs for this item
                # tst_loader batch_size=1, so we take the first item
                iid = str(self.data_handler.tst_dict["iid"][i])
                verified_ids = self.item_vocab.get(iid, set())

                # Build logit processor for this item
                logit_processor = ItemConstrainedLogitsProcessor(
                    verified_token_ids=verified_ids,
                    blacklist_token_ids=self.blacklist_ids,
                    positive_bias=15.0,
                    negative_bias=10.0,
                    device=str(device),
                )

                outputs = self.model.generate(
                    user_embed, item_embed, input_text,
                    logits_processor=[logit_processor]
                )

                end_idx = outputs[0].find("[")
                if end_idx != -1:
                    outputs[0] = outputs[0][:end_idx]

                predictions.append(outputs[0])
                references.append(explain[0])

                if i % 10 == 0 and i != 0:
                    print(f"Step [{i}/{len(self.tst_loader)}]")
                    print(f"Generated Explanation: {outputs[0]}")

        with open(self.tst_predictions_path, "wb") as f:
            pickle.dump(predictions, f)
        with open(self.tst_references_path, "wb") as f:
            pickle.dump(references, f)

        print(f"Saved {len(predictions)} predictions to {self.tst_predictions_path}")


def main():
    sample = XRecBiased()
    print("Generating biased explanations...")
    sample.evaluate()


if __name__ == "__main__":
    main()