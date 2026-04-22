import torch
import pickle
import random
from torch.utils.data import Dataset, DataLoader
from utils.parse import args
from typing import List


class TextDataset(Dataset):
    def __init__(self, input_text: List[str]):
        self.input_text = input_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        return self.input_text[idx]


class DataHandler:
    def __init__(self):
        if args.dataset == "amazon":
            self.system_prompt = "Explain why the user would buy with the book within 50 words."
            self.item = "book"
        elif args.dataset == "yelp" or args.dataset == "google":
            self.system_prompt = "Explain why the user would enjoy the business within 50 words."
            self.item = "business"
        user_path = f"./data/{args.dataset}/user_emb.pkl"
        item_path = f"./data/{args.dataset}/item_emb.pkl"
        with open(user_path, "rb") as file:
            self.user_emb = pickle.load(file)
        with open(item_path, "rb") as file:
            self.item_emb = pickle.load(file)

    def _subsample(self, records, subset_size):
        if subset_size is None or subset_size >= len(records):
            return records
        rng = random.Random(args.subset_seed)
        picked_indices = sorted(rng.sample(range(len(records)), subset_size))
        return [records[idx] for idx in picked_indices]

    def _build_record(self, data_dict, idx, include_explanation):
        record = {
            "uid": int(data_dict["uid"][idx]),
            "iid": int(data_dict["iid"][idx]),
            "title": data_dict["title"][idx],
            "item_summary": data_dict["item_summary"][idx],
            "user_embed": self.user_emb[data_dict["uid"][idx]],
            "item_embed": self.item_emb[data_dict["iid"][idx]],
        }
        user_message = (
            f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> "
            f"{self.item} name: {data_dict['title'][idx]} user profile: "
            f"{data_dict['user_summary'][idx]} {self.item} profile: "
            f"{data_dict['item_summary'][idx]} <EXPLAIN_POS>"
        )
        if include_explanation:
            user_message = f"{user_message} {data_dict['explanation'][idx]}"
        record["input_text"] = (
            f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
        )
        if "explanation" in data_dict:
            record["explain"] = data_dict["explanation"][idx]
        return record

    def load_data(self):
        # load data from data_loaders in data
        with open(f"./data/{args.dataset}/trn.pkl", "rb") as file:
            trn_data = pickle.load(file)
        with open(f"./data/{args.dataset}/val.pkl", "rb") as file:
            val_data = pickle.load(file)
        with open(f"./data/{args.dataset}/tst.pkl", "rb") as file:
            tst_data = pickle.load(file)

        # convert data into dictionary
        trn_dict = trn_data.to_dict("list")
        val_dict = val_data.to_dict("list")
        tst_dict = tst_data.to_dict("list")

        # combine all information input input string
        trn_input = []
        val_input = []
        tst_input = []
        for i in range(len(trn_dict["uid"])):
            trn_input.append(self._build_record(trn_dict, i, include_explanation=True))
        for i in range(len(val_dict["uid"])):
            val_input.append(self._build_record(val_dict, i, include_explanation=False))
        for i in range(len(tst_dict["uid"])):
            tst_input.append(self._build_record(tst_dict, i, include_explanation=False))

        trn_input = self._subsample(trn_input, args.train_subset_size)
        val_input = self._subsample(val_input, args.val_subset_size)
        tst_input = self._subsample(tst_input, args.test_subset_size)

        # load training batch
        trn_dataset = TextDataset(trn_input)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)

        # load validation batch
        val_dataset = TextDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # load testing batch
        tst_dataset = TextDataset(tst_input)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False)

        return trn_loader, val_loader, tst_loader
