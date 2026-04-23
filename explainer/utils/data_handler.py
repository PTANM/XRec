import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.parse import args
from typing import List


class TextDataset(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx]


class DataHandler:
    def __init__(self):

        if args.dataset == "amazon":
            self.system_prompt = "Explain why the user would buy with the book within 50 words."
            self.item = "book"
        elif args.dataset in ["yelp", "google"]:
            self.system_prompt = "Explain why the user would enjoy the business within 50 words."
            self.item = "business"

        with open(f"./data/{args.dataset}/user_emb.pkl", "rb") as f:
            self.user_emb = pickle.load(f)

        with open(f"./data/{args.dataset}/item_emb.pkl", "rb") as f:
            self.item_emb = pickle.load(f)

    def load_data(self):

        with open(f"./data/{args.dataset}/trn.pkl", "rb") as f:
            trn_data = pickle.load(f)
        with open(f"./data/{args.dataset}/val.pkl", "rb") as f:
            val_data = pickle.load(f)
        with open(f"./data/{args.dataset}/tst.pkl", "rb") as f:
            tst_data = pickle.load(f)

        trn_dict = trn_data.to_dict("list")
        val_dict = val_data.to_dict("list")
        tst_dict = tst_data.to_dict("list")

        self.tst_dict = tst_dict

        trn_input, val_input, tst_input = [], [], []

        for i in range(len(trn_dict["uid"])):
            msg = (
                f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> "
                f"{self.item} name: {trn_dict['title'][i]} "
                f"user profile: {trn_dict['user_summary'][i]} "
                f"{self.item} profile: {trn_dict['item_summary'][i]} "
                f"<EXPLAIN_POS> {trn_dict['explanation'][i]}"
            )

            trn_input.append((
                self.user_emb[trn_dict["uid"][i]],
                self.item_emb[trn_dict["iid"][i]],
                f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{msg}[/INST]"
            ))

        for i in range(len(val_dict["uid"])):
            msg = (
                f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> "
                f"{self.item} name: {val_dict['title'][i]} "
                f"user profile: {val_dict['user_summary'][i]} "
                f"{self.item} profile: {val_dict['item_summary'][i]} "
                f"<EXPLAIN_POS>"
            )

            val_input.append((
                self.user_emb[val_dict["uid"][i]],
                self.item_emb[val_dict["iid"][i]],
                f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{msg}[/INST]",
                val_dict["explanation"][i],
            ))

        for i in range(len(tst_dict["uid"])):
            msg = (
                f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> "
                f"{self.item} name: {tst_dict['title'][i]} "
                f"user profile: {tst_dict['user_summary'][i]} "
                f"{self.item} profile: {tst_dict['item_summary'][i]} "
                f"<EXPLAIN_POS>"
            )

            tst_input.append((
                self.user_emb[tst_dict["uid"][i]],
                self.item_emb[tst_dict["iid"][i]],
                f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{msg}[/INST]",
                tst_dict["explanation"][i],
            ))

        # ---------------- SPEED-OPTIMIZED LOADERS ----------------
        num_workers = min(8, args.batch_size if args.batch_size > 1 else 1)

        trn_loader = DataLoader(
            TextDataset(trn_input),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            TextDataset(val_input),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        tst_loader = DataLoader(
            TextDataset(tst_input),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        return trn_loader, val_loader, tst_loader