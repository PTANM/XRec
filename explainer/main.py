import json
import os
import pickle
import re

import torch

from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args


def clean_generation(text):
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[-1]
    if "<EXPLAIN_POS>" in text:
        text = text.split("<EXPLAIN_POS>", 1)[-1]
    if "</s>" in text:
        text = text.split("</s>", 1)[0]
    return re.sub(r"\s+", " ", text).strip()


class XRec:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        print(f"adapter_type: {args.adapter_type}")
        self.run_tag = args.checkpoint_tag or args.adapter_type
        self.model = Explainer()
        self.data_handler = DataHandler()
        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()

        self.pretrained_user_embedding_converter_path = (
            f"./data/{args.dataset}/user_converter.pkl"
        )
        self.pretrained_item_embedding_converter_path = (
            f"./data/{args.dataset}/item_converter.pkl"
        )
        self.user_embedding_converter_path = (
            f"./data/{args.dataset}/user_converter_{self.run_tag}.pkl"
        )
        self.item_embedding_converter_path = (
            f"./data/{args.dataset}/item_converter_{self.run_tag}.pkl"
        )
        self.tst_predictions_path = f"./data/{args.dataset}/tst_pred_{self.run_tag}.pkl"
        self.tst_references_path = f"./data/{args.dataset}/tst_ref_{self.run_tag}.pkl"
        self.tst_results_path = f"./data/{args.dataset}/tst_results_{self.run_tag}.jsonl"

    def _checkpoint_paths(self, use_pretrained=False):
        if use_pretrained:
            if args.adapter_type != "moe":
                raise ValueError(
                    "The provided pretrained adapter checkpoints are only compatible with the MoE adapter."
                )
            return (
                self.pretrained_user_embedding_converter_path,
                self.pretrained_item_embedding_converter_path,
            )
        return self.user_embedding_converter_path, self.item_embedding_converter_path

    def save_adapters(self):
        torch.save(
            self.model.user_embedding_converter.state_dict(),
            self.user_embedding_converter_path,
        )
        torch.save(
            self.model.item_embedding_converter.state_dict(),
            self.item_embedding_converter_path,
        )
        print(f"Saved model to {self.user_embedding_converter_path}")
        print(f"Saved model to {self.item_embedding_converter_path}")

    def load_adapters(self, use_pretrained=False):
        user_path, item_path = self._checkpoint_paths(use_pretrained=use_pretrained)
        if not os.path.exists(user_path) or not os.path.exists(item_path):
            raise FileNotFoundError(
                f"Adapter checkpoints not found: {user_path} and {item_path}"
            )
        map_location = self.model.adapter_device
        self.model.user_embedding_converter.load_state_dict(
            torch.load(user_path, map_location=map_location)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(item_path, map_location=map_location)
        )
        print(f"Loaded model from {user_path}")
        print(f"Loaded model from {item_path}")

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.adapter_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        total_steps = len(self.trn_loader)
        for epoch in range(args.epochs):
            total_loss = 0.0
            self.model.train()
            for step, batch in enumerate(self.trn_loader, start=1):
                user_embed = batch["user_embed"]
                item_embed = batch["item_embed"]
                input_text = batch["input_text"]

                optimizer.zero_grad()
                input_ids, outputs, explain_pos_position = self.model.forward(
                    user_embed, item_embed, input_text
                )
                loss = self.model.loss(input_ids, outputs, explain_pos_position)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if step % 100 == 0 or step == total_steps:
                    print(
                        f"Epoch [{epoch + 1}/{args.epochs}], "
                        f"Step [{step}/{total_steps}], Loss: {loss.item():.4f}"
                    )

            epoch_loss = total_loss / max(total_steps, 1)
            print(f"Epoch [{epoch + 1}/{args.epochs}], Mean Loss: {epoch_loss:.4f}")
            self.save_adapters()

    def evaluate(self):
        self.load_adapters(use_pretrained=args.load_pretrained_adapter)
        self.model.eval()

        predictions = []
        references = []
        results = []

        with torch.no_grad():
            total_steps = len(self.tst_loader)
            for step, batch in enumerate(self.tst_loader, start=1):
                user_embed = batch["user_embed"]
                item_embed = batch["item_embed"]
                input_text = batch["input_text"]

                outputs = self.model.generate(user_embed, item_embed, input_text)
                batch_size = len(outputs)
                for idx in range(batch_size):
                    prediction = clean_generation(outputs[idx])
                    reference = batch["explain"][idx]
                    uid = int(batch["uid"][idx])
                    iid = int(batch["iid"][idx])
                    title = batch["title"][idx]
                    item_summary = batch["item_summary"][idx]
                    metadata_text = f"{title}. {item_summary}".strip()

                    predictions.append(prediction)
                    references.append(reference)
                    results.append(
                        {
                            "uid": uid,
                            "iid": iid,
                            "title": title,
                            "item_summary": item_summary,
                            "item_metadata_text": metadata_text,
                            "prediction": prediction,
                            "reference": reference,
                            "adapter_type": args.adapter_type,
                            "run_tag": self.run_tag,
                        }
                    )

                if step % 10 == 0 or step == total_steps:
                    print(f"Step [{step}/{total_steps}]")
                    print(f"Generated Explanation: {predictions[-1]}")

        with open(self.tst_predictions_path, "wb") as file:
            pickle.dump(predictions, file)
        with open(self.tst_references_path, "wb") as file:
            pickle.dump(references, file)
        with open(self.tst_results_path, "w") as file:
            for row in results:
                file.write(json.dumps(row) + "\n")

        print(f"Saved predictions to {self.tst_predictions_path}")
        print(f"Saved references to {self.tst_references_path}")
        print(f"Saved structured results to {self.tst_results_path}")


def main():
    sample = XRec()
    if args.mode == "finetune":
        print("Finetune model...")
        sample.train()
    elif args.mode == "generate":
        print("Generating explanations...")
        sample.evaluate()


if __name__ == "__main__":
    main()
