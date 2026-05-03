# Counterfactual Explanation Pipeline for XRec

This README documents our counterfactual extension to XRec, which answers the question: *"What item was almost recommended, and what small change would have made the system recommend it?"*

The pipeline has four stages:
1. **Counterfactual Data Generation** — GCD pipeline (`GCD.py`)
2. **LLM-as-Judge Labelling** — GPT-4o evaluation of base explanations
3. **LoRA Fine-Tuning** — Adapt XRec to generate bridge-aware explanations (`finetune_cf.py`)
4. **Evaluation** — Before/after comparison notebooks and scripts

---

## Repository Layout (counterfactual/)

```
counterfactual/
├── GCD.py                                    # Step 1 — Geometric Counterfactual Discovery
├── finetune_cf.py                            # Step 3 — LoRA fine-tuning script
├── generate_cf_explanations.py               # Post-FT — Generate explanations with fine-tuned model
├── explainer.py                              # XRec model wrapper used by the above scripts
│
├── run_gcd.sh                                # SLURM launcher for GCD.py
├── run_finetune.sh                           # SLURM launcher for finetune_cf.py (full run)
├── run_finetune_test.sh                      # SLURM launcher for finetune_cf.py (smoke test, 20 records)
├── run_generate.sh                           # SLURM launcher for generate_cf_explanations.py
│
├── counterfactual_training_dataset.json      # Raw GCD output (base XRec explanations, ~5 k records)
├── counterfactual_training_dataset_amazon.json  # Amazon-specific GCD output
├── counterfactual_training_dataset_evaluated.csv   # GPT-4o labelled dataset (Correct / Partial / Incorrect)
├── counterfactual_training_dataset_evaluated.xlsx  # Same, Excel version
├── counterfactual_finetuned_results.json     # Post-FT model explanations (used for evaluation)
│
├── counterfactual_evaluation.ipynb           # Step 4a — Before vs. After evaluation notebook
├── metrics_eval_xrec.ipynb                   # Step 4b — BERTScore / GPT Score vs. GPT-4o reference
│
├── amazon/                                   # Data symlink (same structure as data/amazon/)
├── data/                                     # Full dataset directories (amazon / google / yelp)
│   └── amazon/
│       ├── user_emb.pkl                      # Pre-trained LightGCN user embeddings
│       ├── item_emb.pkl                      # Pre-trained LightGCN item embeddings
│       ├── user_profile.json                 # GPT-generated user profiles
│       ├── item_profile.json                 # GPT-generated item profiles
│       ├── para_dict.pickle                  # Dataset meta (user_num, item_num, …)
│       ├── total_trn.csv                     # Training interactions (user, item)
│       ├── trn/val/tst.pkl                   # Split interaction records
│       ├── tst_pred.pkl                      # Base XRec generated explanations
│       └── tst_ref.pkl                       # Ground truth explanations
│
├── outputs/
│   └── cf_lora/
│       ├── epoch_1/                          # Checkpoint after epoch 1
│       │   ├── lora_weights.pt
│       │   ├── user_converter.pkl
│       │   └── item_converter.pkl
│       ├── epoch_2/
│       └── epoch_3/                          # Best checkpoint (used for generation)
│
└── explainer/                                # Copy of the base XRec explainer module
    ├── main.py
    └── models/
```

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install openpyxl          # needed by finetune_cf.py to read the .csv evaluation file
```

Set your **Hugging Face token** (required to download Llama-2):
```bash
export HF_TOKEN=<your_hf_token>
```

---

## Step 1 — Counterfactual Data Generation (`GCD.py`)

**What it does:**
- Loads pre-trained LightGCN embeddings (`user_emb.pkl`, `item_emb.pkl`).
- For each user, ranks all unseen items by dot-product score and selects the item at rank K+1 as the **near-miss target**.
- Finds the top-3 **bridge items** that are jointly similar to the user and the target.
- Perturbs the user embedding by increasing α until the target enters Top-K, recording the minimum α* needed.
- Feeds the perturbed embedding + profiles into XRec to generate a **base counterfactual explanation**.
- Outputs ~5,668 records to `counterfactual_training_dataset.json`.

**Key configuration (top of `GCD.py`):**
| Variable | Default | Description |
|---|---|---|
| `DATASET` | `"amazon"` | Dataset name (`amazon` / `google` / `yelp`) |
| `TOP_K` | `10` | Size of the recommendation list |
| `NUM_BRIDGES` | `3` | Number of bridge items to find |
| `TARGET_STRATEGY` | `"near_miss"` | How to pick the target item |

**Run locally:**
```bash
cd counterfactual/
python GCD.py
```

**Run on SLURM (A100, ~15 h):**
```bash
cd counterfactual/
sbatch run_gcd.sh
# logs → logs/gcd_<jobid>.out
```

**Output files:**
| File | Description |
|---|---|
| `counterfactual_training_dataset.json` | One JSON object per user: `user_id`, `target_item`, `bridge_items`, `alpha_star`, `llama_explanation` |
| `counterfactual_training_dataset_amazon.json` | Amazon-specific copy |

---

## Step 2 — LLM-as-Judge Labelling (GPT-4o, manual/notebook step)

**What it does:**
- Each record from Step 1 is sent to GPT-4o with the user profile, target item, bridge items, and α*.
- GPT-4o outputs:
  - A high-quality **reference explanation** (`chatgpt_reference`) that names bridge items.
  - A **quality label**: `Correct`, `Partially Correct`, or `Incorrect`.

**Processed data files:**
| File | Description |
|---|---|
| `counterfactual_training_dataset_evaluated.csv` | GCD records enriched with `chatgpt_response`, `chatgpt_reference`, quality label |
| `counterfactual_training_dataset_evaluated.xlsx` | Same in Excel format |

The fine-tuning step (`finetune_cf.py`) reads `counterfactual_training_dataset_evaluated.csv` and trains **only on records labelled `Correct`** (~1,777 records).

---

## Step 3 — LoRA Fine-Tuning (`finetune_cf.py`)

**What it does:**
- Loads the evaluated CSV from Step 2.
- Injects LoRA adapters (rank 16, α=32) into `q_proj` and `v_proj` of the Llama-2 backbone.
- Keeps the MoE embedding converters (`user_embedding_converter`, `item_embedding_converter`) trainable; freezes everything else.
- Trains for 3 epochs with cross-entropy loss on explanation tokens only.
- Saves `lora_weights.pt`, `user_converter.pkl`, `item_converter.pkl` after every epoch.

**Key configuration (top of `finetune_cf.py`):**
| Variable | Default | Description |
|---|---|---|
| `DATASET` | `"amazon"` | Dataset name (`amazon` / `google` / `yelp`) |
| `CF_DATA_PATH` | `counterfactual_training_dataset_evaluated.csv` | Labelled training data (GPT-4o evaluated CSV) |
| `OUTPUT_DIR` | `outputs/cf_lora/` | Checkpoint save directory |
| `BATCH_SIZE` | `4` | Per-GPU batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps (effective batch = 4 × 4 = 16) |
| `LR` | `2e-4` | AdamW learning rate |
| `EPOCHS` | `3` | Number of training epochs |
| `MAX_LENGTH` | `512` | Maximum prompt + explanation token length |
| `SAVE_EVERY` | `200` | Save a mid-epoch checkpoint every N optimizer steps |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA scaling factor (alpha / r = 2.0) |
| `LORA_DROPOUT` | `0.05` | Dropout applied inside LoRA adapters |
| `LORA_TARGETS` | `["q_proj", "v_proj"]` | LLaMA attention layers that receive LoRA |

**Run locally (full training):**
```bash
cd counterfactual/
python finetune_cf.py
```

**Smoke test (20 records, 1 epoch — quick sanity check):**
```bash
cd counterfactual/
python finetune_cf.py --test
```

**Run on SLURM (A100, ~2 h):**
```bash
cd counterfactual/
sbatch run_finetune.sh                    # full training
sbatch run_finetune_test.sh               # smoke test (~15 min)
# logs → logs/finetune_<jobid>.out
```

**Resume from latest checkpoint:**
```bash
python finetune_cf.py --resume
```

**Output files (under `outputs/cf_lora/`):**
| File | Description |
|---|---|
| `epoch_N/lora_weights.pt` | LoRA delta weights for epoch N |
| `epoch_N/user_converter.pkl` | MoE user embedding adapter weights |
| `epoch_N/item_converter.pkl` | MoE item embedding adapter weights |

---

## Step 3b — Post-FT Explanation Generation (`generate_cf_explanations.py`)

After fine-tuning, generate explanations with the fine-tuned model over the full evaluated dataset:

```bash
cd counterfactual/
python generate_cf_explanations.py \
    --lora_dir outputs/cf_lora/epoch_3 \
    --output  counterfactual_finetuned_results.json
```

**Run on SLURM (A100, ~6 h):**
```bash
sbatch run_generate.sh
# logs → logs/generate_<jobid>.out
```

**Arguments:**
| Flag | Default | Description |
|---|---|---|
| `--lora_dir` | `outputs/cf_lora/epoch_3` | Path to checkpoint directory |
| `--output` | `counterfactual_finetuned_results.json` | Output file path |
| `--limit` | _(all records)_ | Optional: evaluate only the first N records |

**Output file:**
| File | Description |
|---|---|
| `counterfactual_finetuned_results.json` | One record per user with `llama_explanation` (fine-tuned), `chatgpt_reference`, bridge items, α* |

---

## Step 4 — Evaluation

### 4a. Before vs. After Counterfactual Evaluation
**File:** `counterfactual/counterfactual_evaluation.ipynb`

Open in Jupyter and run all cells. The notebook reads:
- **Before:** `counterfactual_training_dataset.json` → field `llama_explanation`
- **After:** `counterfactual_finetuned_results.json` → field `llama_explanation`
- **Reference:** `counterfactual_finetuned_results.json` → field `chatgpt_reference`

**Metrics computed:**
| Metric | What it measures |
|---|---|
| **Basic Stats** | Avg. output length (tokens), template fallback rate |
| **BMR** (Bridge Mention Rate) | Fraction of outputs that name ≥1 bridge item |
| **USR** (Unique Sentence Rate) | Fraction of distinct outputs (personalisation) |
| **Alpha-Sentiment Correlation** | Pearson r between α* and VADER sentiment score |
| **Factual Precision** | Fraction of reference content-words found in the explanation |
| **BARTScore** | log P(explanation \| reference) scored by **BART-large-CNN** (`facebook/bart-large-cnn`); lower (more negative) is worse |

```bash
cd counterfactual/
jupyter notebook counterfactual_evaluation.ipynb
```

### 4b. Standard XRec Metrics (BERTScore / GPT Score)
**File:** `counterfactual/metrics_eval_xrec.ipynb`

Computes BERTScore F1/Precision/Recall and GPT Score against the GPT-4o reference explanation.

```bash
cd counterfactual/
jupyter notebook metrics_eval_xrec.ipynb
```

---

## End-to-End Pipeline Summary

```
[data/amazon/]          [Step 1]             [Step 2]
user_emb.pkl  ──────►  GCD.py  ──────►  GPT-4o labelling
item_emb.pkl            │                      │
profiles                ▼                      ▼
                counterfactual_        counterfactual_
                training_dataset       training_dataset
                .json                  _evaluated.csv
                                              │
                                       [Step 3]
                                       finetune_cf.py
                                              │
                                       outputs/cf_lora/epoch_3/
                                              │
                                  [Step 3b]   ▼
                                  generate_cf_explanations.py
                                              │
                                  counterfactual_finetuned_results.json
                                              │
                              ┌───────────────┴──────────────────┐
                         [Step 4a]                          [Step 4b]
                 counterfactual_evaluation.ipynb    metrics_eval_xrec.ipynb
                 (BMR, USR, Alpha-Sentiment, …)     (BERTScore, GPT Score)
```

---

## Quick-Start (Local, no SLURM)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install openpyxl
export HF_TOKEN=<your_token>

# 2. Run GCD pipeline
cd counterfactual/
python GCD.py

# 3. (Manual) Label with GPT-4o — produces counterfactual_training_dataset_evaluated.csv

# 4. Fine-tune (smoke test first)
python finetune_cf.py --test          # 20 records, 1 epoch
python finetune_cf.py                 # full run (3 epochs)

# 5. Generate post-FT explanations
python generate_cf_explanations.py --lora_dir outputs/cf_lora/epoch_3

# 6. Evaluate
jupyter notebook counterfactual_evaluation.ipynb
jupyter notebook metrics_eval_xrec.ipynb
```
