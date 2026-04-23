import argparse
import csv
import json
import os
import pickle
import sys
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
import pandas as pd
import traceback
import huggingface_hub

# --- Global Configuration ---
ROOT = Path('./') 
DATASET = "amazon" 
DATA_DIR = ROOT / DATASET
TOP_K = 10
NUM_BRIDGES = 3
TARGET_STRATEGY = "near_miss" 

torch.set_grad_enabled(False)

print(f"Root data directory: {ROOT.resolve()}")
print(f"Dataset directory: {DATA_DIR.resolve()}")

# Determine the device for general torch operations
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU for torch operations: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU for torch operations: {device}")

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_completion_text(raw: str) -> Dict[str, str]:
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"summarization": raw}

def load_profiles(path: Path, id_key: str) -> Dict[int, Dict[str, str]]:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = parse_completion_text(row.get("completion", ""))
            records[int(row[id_key])] = {
                "summary": text.get("summarization", ""),
                "reasoning": text.get("reasoning", ""),
            }
    return records

def load_seen_items(*csv_paths: Path) -> Dict[int, set]:
    seen = defaultdict(set)
    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen[int(row["user"])].add(int(row["item"]))
    return dict(seen)

def load_titles_from_pair_pickles(*pickle_paths: Path) -> Dict[int, str]:
    titles = {}
    for path in pickle_paths:
        rows = load_pickle(path)
        if hasattr(rows, "to_dict"):
            rows = rows.to_dict("records")
        for row in rows:
            iid = int(row["iid"])
            title = row.get("title")
            if title and iid not in titles:
                titles[iid] = title
    return titles

# Load embeddings and move them to the determined device
user_emb = load_pickle(DATA_DIR / "user_emb.pkl").to(device)
item_emb = load_pickle(DATA_DIR / "item_emb.pkl").to(device)
para_dict = load_pickle(DATA_DIR / "para_dict.pickle")

user_profiles = load_profiles(DATA_DIR / "user_profile.json", "uid")
item_profiles = load_profiles(DATA_DIR / "item_profile.json", "iid")
seen_items = load_seen_items(DATA_DIR / "total_trn.csv")
item_titles = load_titles_from_pair_pickles(
    DATA_DIR / "trn.pkl",
    DATA_DIR / "val.pkl",
    DATA_DIR / "tst.pkl",
)

assert user_emb.shape[0] == para_dict["user_num"]
assert item_emb.shape[0] == para_dict["item_num"]

print(f"User embeddings: {tuple(user_emb.shape)}")
print(f"Item embeddings: {tuple(item_emb.shape)}")
print(f"Loaded {len(user_profiles)} user profiles and {len(item_profiles)} item profiles")
print(f"Loaded seen-item map for {len(seen_items)} users")
print(f"Recovered titles for {len(item_titles)} items from train/val/test pairs")

def item_name(iid: int) -> str:
    return item_titles.get(iid, f"item_{iid}")

def item_summary(iid: int) -> str:
    return item_profiles.get(iid, {}).get("summary", "")

def user_summary(uid: int) -> str:
    return user_profiles.get(uid, {}).get("summary", "")

def show_item(iid: int):
    print(f"Item {iid}: {item_name(iid)}")
    print(item_summary(iid)[:400] or "<no summary>")

def show_user(uid: int):
    print(f"User {uid}")
    print(user_summary(uid)[:500] or "<no summary>")

class TargetSelector:
    def __init__(self, user_emb: torch.Tensor, item_emb: torch.Tensor, seen_items: Optional[Dict[int, set]] = None, device: torch.device = None):
        self.user_emb = user_emb.float()
        self.item_emb = item_emb.float()
        self.seen_items = seen_items or {}
        self.num_users = user_emb.shape[0]
        self.num_items = item_emb.shape[0]
        self.device = device if device is not None else user_emb.device # Use the device of the embeddings if not explicitly passed

    def score_all_items(self, user_id: int) -> torch.Tensor:
        return self.user_emb[user_id] @ self.item_emb.T

    def candidate_mask(self, user_id: int) -> torch.Tensor:
        mask = torch.ones(self.num_items, dtype=torch.bool, device=self.device) # Specify device
        for iid in self.seen_items.get(user_id, set()):
            mask[iid] = False
        return mask

    def ranked_candidates(self, user_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_all_items(user_id)
        mask = self.candidate_mask(user_id)
        # Ensure torch.arange is also on the correct device
        candidate_ids_all = torch.arange(self.num_items, device=self.device) # Specify device
        candidate_ids = candidate_ids_all[mask]
        candidate_scores = scores[mask]
        sorted_idx = torch.argsort(candidate_scores, descending=True)
        return candidate_ids[sorted_idx], candidate_scores[sorted_idx]

    def recommended_items(self, user_id: int, top_k: int = 10) -> List[int]:
        ranked_ids, _ = self.ranked_candidates(user_id)
        return ranked_ids[:top_k].tolist()

    def select_target(self, user_id: int, top_k: int = 10, near_miss_window: int = 5, strategy: str = "near_miss") -> Dict:
        ranked_ids, ranked_scores = self.ranked_candidates(user_id)
        recommended = ranked_ids[:top_k].tolist()

        if strategy == "near_miss":
            start = top_k
            end = min(top_k + near_miss_window, len(ranked_ids))
            target_idx = start
        elif strategy == "highest_non_recommended":
            target_idx = top_k
        elif strategy == "random_non_recommended":
            # Ensure torch.randint is on the correct device
            target_idx = int(torch.randint(low=top_k, high=len(ranked_ids), size=(1,), device=self.device).item())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        target_item_id = int(ranked_ids[target_idx].item())
        target_score = float(ranked_scores[target_idx].item())
        threshold = float(ranked_scores[top_k - 1].item())

        return {
            "user_id": user_id,
            "recommended": recommended,
            "target_item_id": target_item_id,
            "target_score": target_score,
            "recommendation_threshold": threshold,
            "gap_to_threshold": threshold - target_score,
            "ranked_candidate_ids": ranked_ids,
            "ranked_candidate_scores": ranked_scores,
        }

class BridgeFinder:
    def __init__(self, user_emb: torch.Tensor, item_emb: torch.Tensor, seen_items: Optional[Dict[int, set]] = None, device: torch.device = None):
        self.user_emb = user_emb.float()
        self.item_emb = item_emb.float()
        self.seen_items = seen_items or {}
        self.device = device if device is not None else user_emb.device
        self.user_norms = self.user_emb / torch.norm(self.user_emb, dim=1, keepdim=True).clamp_min(1e-12)
        self.item_norms = self.item_emb / torch.norm(self.item_emb, dim=1, keepdim=True).clamp_min(1e-12)

    def find(self, user_id: int, target_item_id: int, k: int = 5, exclude_recommended: Optional[Sequence[int]] = None) -> List[Dict]:
        user_vec = self.user_norms[user_id]
        target_vec = self.item_norms[target_item_id]

        sim_to_user = self.item_norms @ user_vec
        sim_to_target = self.item_norms @ target_vec
        bridge_scores = sim_to_user * sim_to_target

        bridge_scores[target_item_id] = -torch.inf
        for iid in self.seen_items.get(user_id, set()):
            bridge_scores[iid] = -torch.inf
        for iid in exclude_recommended or []:
            bridge_scores[iid] = -torch.inf

        values, indices = torch.topk(bridge_scores, k=min(k, bridge_scores.shape[0]))
        results = []
        for score, iid in zip(values.tolist(), indices.tolist()):
            if score == float("-inf"):
                continue
            results.append({
                "item_id": int(iid),
                "bridge_score": float(score),
                "sim_to_user": float(sim_to_user[iid].item()),
                "sim_to_target": float(sim_to_target[iid].item()),
            })
        return results

class PerturbationEngine:
    def __init__(self, item_emb: torch.Tensor, device: torch.device = None):
        self.item_emb = item_emb.float()
        self.device = device if device is not None else item_emb.device

    def bridge_vector(self, bridge_item_ids: Sequence[int]) -> torch.Tensor:
        return self.item_emb[list(bridge_item_ids)].mean(dim=0)

    def perturb(self, user_vector: torch.Tensor, bridge_item_ids: Sequence[int], alpha: float) -> torch.Tensor:
        bridge_vec = self.bridge_vector(bridge_item_ids)
        return user_vector + alpha * (bridge_vec - user_vector)


class Verifier:
    def __init__(self, item_emb: torch.Tensor, seen_items: Optional[Dict[int, set]] = None, device: torch.device = None):
        self.item_emb = item_emb.float()
        self.num_items = item_emb.shape[0]
        self.seen_items = seen_items or {}
        self.device = device if device is not None else item_emb.device # Use the device of the embeddings if not explicitly passed

    def score_candidates(self, user_id: int, user_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = user_vector.float() @ self.item_emb.T
        mask = torch.ones(self.num_items, dtype=torch.bool, device=self.device) # Specify device
        for iid in self.seen_items.get(user_id, set()):
            mask[iid] = False
        # Ensure torch.arange is on the correct device
        candidate_ids_all = torch.arange(self.num_items, device=self.device) # Specify device
        candidate_ids = candidate_ids_all[mask]
        candidate_scores = scores[mask]
        sorted_idx = torch.argsort(candidate_scores, descending=True)
        return candidate_ids[sorted_idx], candidate_scores[sorted_idx]

    def verify(self, user_id: int, user_vector: torch.Tensor, target_item_id: int, top_k: int = 10) -> Dict:
        ranked_ids, ranked_scores = self.score_candidates(user_id, user_vector)
        top_items = ranked_ids[:top_k].tolist()
        found_rank = None
        found_score = None
        # Ensure `target_item_id` is a tensor on the correct device for comparison
        target_item_id_tensor = torch.tensor(target_item_id, device=self.device)
        matches = (ranked_ids == target_item_id_tensor).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            idx = int(matches[0].item())
            found_rank = idx + 1
            found_score = float(ranked_scores[idx].item())
        return {
            "is_recommended": target_item_id in top_items,
            "top_items": top_items,
            "target_rank": found_rank,
            "target_score": found_score,
            "recommendation_threshold": float(ranked_scores[top_k - 1].item()),
        }

def search_minimum_alpha(
    user_id: int,
    user_vector: torch.Tensor,
    bridge_item_ids: Sequence[int],
    target_item_id: int,
    perturbation_engine: PerturbationEngine,
    verifier: Verifier,
    top_k: int = 10,
    alphas: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0),
) -> Dict:
    trials = []
    for alpha in alphas:
        perturbed = perturbation_engine.perturb(user_vector, bridge_item_ids, alpha)
        verdict = verifier.verify(user_id, perturbed, target_item_id, top_k=top_k)
        trials.append({
            "alpha": alpha,
            **verdict,
            "perturbed_user_vector": perturbed,
        })
        if verdict["is_recommended"]:
            return {
                "success": True,
                "best_trial": trials[-1],
                "trials": trials,
            }
    return {
        "success": False,
        "best_trial": trials[-1] if trials else None,
        "trials": trials,
    }

def build_counterfactual_prompt(
    dataset: str,
    user_id: int,
    target_item_id: int,
    bridge_item_ids: Sequence[int],
    original_score: float,
    perturbed_score: float,
    top_k: int,
    alpha: float,
) -> str:
    item_word = "book" if dataset == "amazon" else "business"
    system_prompt = (
        f"Task: Explain why a book was originally missed and how 'bridge' books reveal its relevance. "
        f"Logic: The target book seemed too focused on [Target Theme], but the bridge books show it actually contains [User Interest]. "
        f"Structure: 1. Identify the thematic gap. 2. Use the bridge books to link the user's profile to the target book's profile. "
        f"Do not invent interests that are not listed in the user profile. Do not lie about book themes."
    )

    bridge_lines = []
    for iid in bridge_item_ids:
        bridge_lines.append(f"- {item_name(iid)}: {item_summary(iid)}")

    bridge_names_str = ", ".join(item_name(iid) for iid in bridge_item_ids)

    return (
        f"<s>[INST] <<SYS>>\n"
        f"You are a logic engine. Your goal is to explain how the BRIDGE BOOKS prove that the TARGET BOOK fits the USER PROFILE.\n"
        f"<</SYS>>\n\n"
        f"USER PROFILE: {user_summary(user_id)}\n"
        f"TARGET: {item_name(target_item_id)} ({item_summary(target_item_id)})\n"
        f"BRIDGES:\n" + "\n".join(bridge_lines) + "\n"
        f"DATA: <USER_EMBED> <ITEM_EMBED>\n"
        f"[/INST] If the user explored {bridge_names_str}, their preferences would shift toward '{item_name(target_item_id)}'. Specifically, <EXPLAIN_POS>"
    )

def template_counterfactual_explanation(user_id: int, target_item_id: int, bridge_item_ids: Sequence[int], alpha: float, top_k: int = 10) -> str:
    bridge_names = ", ".join(item_name(iid) for iid in bridge_item_ids)
    return (
        f"{item_name(target_item_id)} is not currently in the user's top-{top_k} recommendations because the user's present profile is not yet close enough to that item's themes. "
        f"Items such as {bridge_names} connect the user's current interests to the target. "
        f"If the user showed stronger interest in those bridge items, the user representation would shift toward the target and make it recommendable at alpha={alpha:.2f}."
    )

class XRecCounterfactualExplainer:
    def __init__(self, root: Path, dataset: str):
        self.root = root
        self.dataset = dataset
        self.model = None
        # Determine device once for the explainer model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Explainer will use device: {self.device}")

    def load(self):
        if self.model is not None:
            return self.model

        explainer_path = str(self.root / "explainer")
        if explainer_path not in sys.path:
            sys.path.insert(0, explainer_path)

        # Forcefully remove any existing 'models' package from cache
        for key in list(sys.modules.keys()):
            if key == 'models' or key.startswith('models.'):
                del sys.modules[key]
        importlib.invalidate_caches()

        # type: ignore[reportMissingImports] - resolves Pylance warning for dynamic import
        from models.explainer import Explainer  # noqa: E402

        model = Explainer()
        model.user_embedding_converter.load_state_dict(torch.load(self.root / self.dataset / "user_converter.pkl", map_location=self.device))
        model.item_embedding_converter.load_state_dict(torch.load(self.root / self.dataset / "item_converter.pkl", map_location=self.device))
        model.to(self.device) # Move the entire model to the device
        model.eval()
        self.model = model
        return self.model

    def generate(self, user_vector: torch.Tensor, item_vector: torch.Tensor, prompt: str) -> str:
        model = self.load()
        # Ensure input tensors are on the same device as the model
        output = model.generate(user_vector.unsqueeze(0).to(self.device), item_vector.unsqueeze(0).to(self.device), [prompt])
        return output[0]

def generate_training_dataset(generator, num_users: int) -> list:
    """
    Generate full training dataset by running pipeline over multiple users
    Saves: User, Target, Bridges, Alpha, Explanation all in structured format
    """

    training_records = []
    user_ids = list(range(num_users))

    print(f"Generating training dataset for {num_users} users...")

    for i, user_id in enumerate(tqdm(user_ids, desc="Processing Users")):
        try:
            print(f"\n--- Processing User {user_id} ---")

            # Step 1: Select target
            target_info = selector.select_target(
                user_id,
                top_k=TOP_K,
                near_miss_window=5,
                strategy=TARGET_STRATEGY # Using global TARGET_STRATEGY
            )
            print(f"Target selected: Item {target_info['target_item_id']} (Score gap to top-K: {target_info['gap_to_threshold']:.4f})")

            # Step 2: Find bridges
            bridge_info = bridge_finder.find(
                user_id=user_id,
                target_item_id=target_info["target_item_id"],
                k=NUM_BRIDGES, # Using global NUM_BRIDGES
                exclude_recommended=target_info["recommended"],
            )

            bridge_ids = [row["item_id"] for row in bridge_info]
            print(f"Bridges found: {bridge_ids}")

            # Step 3: Find minimum alpha
            search_result = search_minimum_alpha(
                user_id=user_id,
                user_vector=user_emb[user_id].float(),
                bridge_item_ids=bridge_ids,
                target_item_id=target_info["target_item_id"],
                perturbation_engine=perturbation_engine,
                verifier=verifier,
                top_k=TOP_K,
            )

            if not search_result["success"]:
                print(f"-> Failed to find valid alpha to push target into top-K. Skipping user.")
                continue

            best_trial = search_result["best_trial"]
            print(f"-> Alpha search successful! Minimum alpha required: {best_trial['alpha']:.2f}")

            # Step 4: Generate explanation prompt
            prompt = build_counterfactual_prompt(
                dataset=DATASET,
                user_id=user_id,
                target_item_id=target_info["target_item_id"],
                bridge_item_ids=bridge_ids,
                original_score=target_info["target_score"],
                perturbed_score=best_trial["target_score"],
                top_k=TOP_K,
                alpha=best_trial["alpha"],
            )

            # Step 5: Generate LLaMA explanation
            explanation = ""
            try:
                explanation = generator.generate(
                    user_vector=best_trial["perturbed_user_vector"].float(),
                    item_vector=item_emb[target_info["target_item_id"]].float(),
                    prompt=prompt,
                )
                print(f"-> Explanation generated via LLaMA (Length: {len(explanation)} chars).")
            except Exception as e:
                print(f"-> LLaMA generation failed: {e}. Falling back to template.")
                explanation = template_counterfactual_explanation(
                    user_id=user_id,
                    target_item_id=target_info["target_item_id"],
                    bridge_item_ids=bridge_ids,
                    alpha=best_trial["alpha"],
                    top_k=TOP_K,
                )

            # Create training record
            training_records.append({
                "user_id": user_id,
                "user_profile": user_summary(user_id),
                "target_item_id": target_info["target_item_id"],
                "target_name": item_name(target_info["target_item_id"]),                 "target_profile": item_summary(target_info["target_item_id"]),
                "original_score": target_info["target_score"],
                "recommendation_threshold": target_info["recommendation_threshold"],
                "gap_to_to_threshold": target_info["gap_to_threshold"],
                "bridge_item_ids": bridge_ids,
                "bridge_names": [item_name(iid) for iid in bridge_ids],
                "bridge_profiles": [item_summary(iid) for iid in bridge_ids],
                "minimum_alpha": best_trial["alpha"],
                "final_rank": best_trial["target_rank"],
                "final_score": best_trial["target_score"],
                "counterfactual_prompt": prompt,
                "llama_explanation": explanation,
                "strategy": TARGET_STRATEGY
            })

            # Step 6: Checkpointing at specific milestones
            processed_count = i + 1
            if processed_count in [10, 100] or processed_count % 2000 == 0:
                checkpoint_file = f"counterfactual_training_dataset_checkpoint_{processed_count}.json"
                print(f"\n[Checkpoint] Saving progress at {processed_count} processed users to {checkpoint_file}...")
                with open(checkpoint_file, "w") as f:
                    json.dump(training_records, f, indent=2)

                csv_file = f"counterfactual_training_dataset_checkpoint_{processed_count}.csv"
                pd.DataFrame(training_records).to_csv(csv_file, index=False)

        except Exception as e:
            print(f"-> ❌ Skipping user {user_id} due to pipeline error: {e}")
            traceback.print_exc()
            continue

    return training_records

# --- Main Execution --- 
if __name__ == "__main__":
    # Suppress HF interactive login prompt by using token from environment variable
    try:
        hf_token = os.getenv('HF_TOKEN') # Use environment variable instead
        if hf_token:
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
            os.environ['HF_TOKEN'] = hf_token
        else:
            print("Warning: HF_TOKEN environment variable not found. Hugging Face login skipped.")
    except Exception as e:
        print(f"Warning: Could not automatically login to Hugging Face. Ensure 'HF_TOKEN' is set in your environment. {e}")

    # Initialize the pipeline components locally
    selector = TargetSelector(user_emb, item_emb, seen_items=seen_items, device=device)
    bridge_finder = BridgeFinder(user_emb, item_emb, seen_items=seen_items, device=device)
    perturbation_engine = PerturbationEngine(item_emb, device=device)
    verifier = Verifier(item_emb, seen_items=seen_items, device=device)

    # Initialize Explainer Generator ONCE before the loop
    print("Initializing LLaMA Explainer Generator (this may take a moment)...")
    llama_generator = XRecCounterfactualExplainer(ROOT, DATASET)
    llama_generator.load() # Pre-load the model to avoid doing it inside the loop
    print("Explainer initialized successfully.")

    # Generate and save training dataset
    # The original notebook used `len(user_emb)` for `num_users` which might be very large.
    # For testing, you might want to use a smaller number, e.g., `num_users=100`.
    training_data = generate_training_dataset(generator=llama_generator, num_users=7000) # TEST RUN: 20 users ~10 minutes

    # Save as JSON
    final_json_path = ROOT / f"counterfactual_training_dataset_{DATASET}.json"
    with open(final_json_path, "w") as f:
        json.dump(training_data, f, indent=2)

    # Save as CSV
    df = pd.DataFrame(training_data)
    final_csv_path = ROOT / f"counterfactual_training_dataset_{DATASET}.csv"
    df.to_csv(final_csv_path, index=False)

    print(f"\n Training dataset generated successfully!")
    print(f"   Total valid records: {len(training_data)}")

    if not df.empty:
        print(f"   Average alpha: {df['minimum_alpha'].mean():.4f}")
        # Show sample record
        print("\n Sample Training Record:")
        print(json.dumps(training_data[0], indent=2))
    else:
        print("   No valid records were generated.")

    print(f"   Saved to: {final_json_path.resolve()}")
    print(f"   Saved to: {final_csv_path.resolve()}")
