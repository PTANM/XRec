# MoE vs. 2-Layer MLP Adapter Ablation for XRec

This document describes an ablation study for XRec on the `mlp-test` branch:

**Can a lightweight 2-layer MLP replace the original XRec Mixture-of-Experts (MoE) adapter without degrading explanation quality?**

The pipeline has five stages:

1. **Baseline MoE Generation** — Run the released pretrained XRec MoE adapter on the test split.
2. **MLP Fine-Tuning** — Train a 2-layer MLP adapter while keeping the LLaMA backbone frozen.
3. **MLP Generation** — Generate explanations with the trained MLP adapter.
4. **Evaluation** — Compare MoE and MLP using XRec-style semantic metrics plus grounding-oriented metrics.
5. **Side-by-Side Comparison** — Align the same test examples across two runs and inspect predictions row by row.

---

## Repository Layout

```text
XRec/
├── data/
│   └── amazon/
│       ├── trn.pkl                      # Train split
│       ├── val.pkl                      # Validation split
│       ├── tst.pkl                      # Test split
│       ├── user_emb.pkl                 # LightGCN user embeddings
│       ├── item_emb.pkl                 # LightGCN item embeddings
│       ├── user_converter.pkl           # Released pretrained MoE user adapter
│       ├── item_converter.pkl           # Released pretrained MoE item adapter
│       ├── user_converter_<tag>.pkl     # Saved run-specific user adapter
│       ├── item_converter_<tag>.pkl     # Saved run-specific item adapter
│       ├── tst_pred_<tag>.pkl           # Generated explanations
│       ├── tst_ref_<tag>.pkl            # Reference explanations
│       ├── tst_results_<tag>.jsonl      # Structured per-example results
│       └── comparison_<left>_vs_<right>.csv
│
├── encoder/
│   ├── models/                          # GNN model components
│   ├── utils/                           # Encoder utilities
│   └── train_encoder.py                 # Train LightGCN / derive user-item embeddings
│
├── evaluation/
│   ├── main.py                          # Main evaluation entry point
│   ├── metrics.py                       # LlamaScore, BARTScore, BERTScore, ROUGE, BLEU, factual precision
│   ├── compare_results.py               # Side-by-side CSV comparison for two runs
│   └── system_prompt.txt                # Prompt for optional GPT-style judge scoring
│
├── explainer/
│   ├── main.py                          # Train or generate with MoE / MLP adapters
│   ├── sample.py                        # Preview saved prediction/reference files
│   ├── models/
│   │   ├── explainer.py                 # MoE and MLP adapter definitions + LLaMA wrapper
│   │   └── modeling_explainer.py        # Custom XRec LLaMA backbone
│   └── utils/
│       ├── data_handler.py              # Dataset loading and optional subsampling
│       └── parse.py                     # CLI arguments
│
├── generation/                          # Original XRec profile/explanation generation scripts
├── process/                             # Data processing utilities
├── requirements.txt
└── README.md
```

The files `tst_pred_<tag>.pkl`, `tst_ref_<tag>.pkl`, and `tst_results_<tag>.jsonl` are run-specific generated outputs. The files `user_converter.pkl` and `item_converter.pkl` are the released pretrained MoE adapter checkpoints.

---

## What Changed Relative to Base XRec?

The original XRec model uses MoE adapters to convert LightGCN user and item embeddings into the LLaMA embedding space.

In this ablation, we add a 2-layer MLP adapter option while keeping the frozen LLaMA backbone, prompt construction, and generation pipeline unchanged.

Each 64-dimensional LightGCN embedding is mapped into the 4096-dimensional LLaMA embedding space using:

```text
Linear(64 → 512) → GELU → Dropout → Linear(512 → 4096)
```

The implementation uses two separate MLP embedding converters with the same architecture but independent parameters:

```text
user_embedding_converter
item_embedding_converter
```

The original released MoE baseline remains available through:

```bash
--adapter_type moe
--load_pretrained_adapter
```

The MLP adapter is selected through:

```bash
--adapter_type mlp
--adapter_hidden_size 512
--adapter_dropout 0.2
```

---

## Prerequisites

Install project dependencies:

```bash
pip install -r requirements.txt
```

If running gated LLaMA-2 models, set your Hugging Face token or log in through Hugging Face as required by your environment.

---

## Step 1 — Baseline MoE Generation

### What it does

This step:

- Loads the released pretrained XRec MoE adapter checkpoints:
  - `data/<dataset>/user_converter.pkl`
  - `data/<dataset>/item_converter.pkl`
- Loads the frozen LLaMA backbone.
- Generates explanations on the requested test split or test subset.
- Saves predictions, references, and structured JSONL outputs.

### Run locally

```bash
python explainer/main.py \
  --mode generate \
  --dataset amazon \
  --adapter_type moe \
  --checkpoint_tag baseline_moe \
  --load_pretrained_adapter
```

### Key output files

```text
data/amazon/tst_pred_baseline_moe.pkl
data/amazon/tst_ref_baseline_moe.pkl
data/amazon/tst_results_baseline_moe.jsonl
```

Each line in `tst_results_<tag>.jsonl` contains:

```text
uid
iid
title
item_summary
item_metadata_text
prediction
reference
adapter_type
run_tag
```

For the baseline run used in the main comparison, the corresponding files are:

```text
data/amazon/tst_pred_baseline_moe_6250.pkl
data/amazon/tst_ref_baseline_moe_6250.pkl
data/amazon/tst_results_baseline_moe_6250.jsonl
```

---

## Step 2 — MLP Fine-Tuning

### What it does

This step:

- Replaces the MoE adapter with a 2-layer MLP adapter.
- Keeps the LLaMA backbone frozen.
- Trains only:
  - `user_embedding_converter`
  - `item_embedding_converter`
- Optimizes next-token prediction loss over the explanation portion of the prompt.
- Saves run-specific adapter checkpoints after each epoch.

### Important note on validation

The data pipeline can load and subsample a validation split, but the current training loop does **not** use validation for:

- early stopping,
- checkpoint selection,
- hyperparameter search.

So validation may be loaded, but it is not used for model selection in the current script.

### Run locally

```bash
python explainer/main.py \
  --mode finetune \
  --dataset amazon \
  --adapter_type mlp \
  --checkpoint_tag mlp \
  --epochs 1 \
  --lr 1e-4 \
  --adapter_hidden_size 512 \
  --adapter_dropout 0.2 \
  --batch_size 2
```

### Optional subset controls

```bash
--train_subset_size <N>
--val_subset_size <N>
--test_subset_size <N>
--subset_seed 42
```

### Key output files

```text
data/amazon/user_converter_<tag>.pkl
data/amazon/item_converter_<tag>.pkl
```

For example, if `--checkpoint_tag mlp` is used, the expected adapter checkpoint files are:

```text
data/amazon/user_converter_mlp.pkl
data/amazon/item_converter_mlp.pkl
```

---

## Step 3 — MLP Generation

After training, generate explanations with the trained MLP adapter:

```bash
python explainer/main.py \
  --mode generate \
  --dataset amazon \
  --adapter_type mlp \
  --checkpoint_tag mlp \
  --adapter_hidden_size 512 \
  --adapter_dropout 0.2 \
  --batch_size 2
```

### Key output files

```text
data/amazon/tst_pred_mlp.pkl
data/amazon/tst_ref_mlp.pkl
data/amazon/tst_results_mlp.jsonl
```

For the run used in the main comparison, the corresponding files are:

```text
data/amazon/tst_pred_mlp_50k.pkl
data/amazon/tst_ref_mlp_50k.pkl
data/amazon/tst_results_mlp_50k.jsonl
```

---

## Step 4 — Evaluation

### What it does

The main evaluation script reads `tst_results_<tag>.jsonl`, compares generated explanations against reference explanations, and reports:

```text
llama_score
bart_score
bert_precision
bert_recall
bert_f1
rouge1
rouge2
rougeL
bleu
bleu1
bleu2
bleu3
bleu4
factual_precision
attribute_matches
usr
avg_length
```

The mlp_50k tag used in the results files refers to the MLP adapter trained on a 50,000-example subset of the training split. The baseline_moe_6250 tag is the corresponding MoE baseline run name that was used during experimentation. Although the tag contains 6250, the final evaluation files analyzed contain 3,000 test instances.

Depending on runtime flags, model-heavy scores such as LlamaScore or BARTScore can be skipped. The default structured input path is:

```text
data/<dataset>/tst_results_<results_tag>.jsonl
```

### Important note on BERTScore

The evaluation uses **baseline-rescaled BERTScore**. Because of this, negative BERTScore values are possible. A negative value does not mean the generated text is invalid; it means the score falls below the BERTScore baseline reference point.

### Important note on factual precision

`factual_precision` is a custom metadata-grounded attribute precision metric, not FActScore. It measures the fraction of generated attribute phrases that can be matched back to the target item metadata text.

Because XRec provides processed item titles and summaries rather than the complete raw UCSD Amazon metadata, factual precision is computed against the metadata proxy available in XRec.

### Run locally

```bash
python evaluation/main.py \
  --dataset amazon \
  --results_tag baseline_moe
```

```bash
python evaluation/main.py \
  --dataset amazon \
  --results_tag mlp
```

## Step 5 — Side-by-Side CSV Comparison

To align the same test examples across two runs and compare predictions row by row:

```bash
python evaluation/compare_results.py \
  --dataset amazon \
  --left_tag baseline_moe \
  --right_tag mlp
```

### Output file

```text
data/amazon/comparison_baseline_moe_vs_mlp.csv
```

This CSV includes:

```text
uid
iid
title
item_summary
item_metadata_text
reference
<left_tag>_prediction
<right_tag>_prediction
<left_tag>_adapter_type
<right_tag>_adapter_type
<left_tag>_run_tag
<right_tag>_run_tag
<left_tag>_word_count
<right_tag>_word_count
reference_word_count
present_in_left
present_in_right
```

You can also pass explicit paths if the default filenames are not used.

---

## Suggested End-to-End Local Run

```bash
# 1. Generate explanations with the released pretrained MoE baseline
python explainer/main.py \
  --mode generate \
  --dataset amazon \
  --adapter_type moe \
  --checkpoint_tag baseline_moe \
  --load_pretrained_adapter

# 2. Fine-tune the 2-layer MLP adapter
python explainer/main.py \
  --mode finetune \
  --dataset amazon \
  --adapter_type mlp \
  --checkpoint_tag mlp \
  --epochs 1 \
  --lr 1e-4 \
  --adapter_hidden_size 512 \
  --adapter_dropout 0.2 \
  --batch_size 2

# 3. Generate explanations with the trained MLP adapter
python explainer/main.py \
  --mode generate \
  --dataset amazon \
  --adapter_type mlp \
  --checkpoint_tag mlp \
  --adapter_hidden_size 512 \
  --adapter_dropout 0.2 \
  --batch_size 2

# 4. Evaluate both runs
python evaluation/main.py --dataset amazon --results_tag baseline_moe
python evaluation/main.py --dataset amazon --results_tag mlp

# 5. Create a side-by-side comparison CSV
python evaluation/compare_results.py \
  --dataset amazon \
  --left_tag baseline_moe \
  --right_tag mlp
```

---

## HPRC / SLURM Workflow

For cluster runs, we used a local HPRC script named `run_mlp.slurm`, which executes the MoE-vs-MLP pipeline end to end. This script is useful for reproducing the cluster workflow, but the Python commands above are sufficient for local reproduction.

The SLURM workflow performs the following steps:

1. generate explanations using the released pretrained MoE adapter,
2. fine-tune the 2-layer MLP adapter,
3. generate explanations using the trained MLP adapter,
4. evaluate both runs, and
5. write a side-by-side comparison CSV.

The script is controlled through SLURM environment variables. The most important ones are:

```text
TRAIN_SUBSET          Number of training examples used for MLP fine-tuning
VAL_SUBSET            Number of validation examples loaded
TEST_SUBSET           Number of test examples used for generation/evaluation
MLP_TAG               Output tag for the MLP run
BASELINE_TAG          Output tag for the MoE baseline run
EPOCHS                Number of MLP fine-tuning epochs
BATCH_SIZE            Batch size
LR                    Learning rate
ADAPTER_HIDDEN_SIZE   Hidden size of the MLP adapter
MODEL_NAME            LLaMA model name
LOAD_IN_8BIT          Whether to load the LLaMA model in 8-bit mode
```

For the main reported experiment, we used:

```bash
sbatch --export=ALL,TRAIN_SUBSET=50000,VAL_SUBSET=6250,TEST_SUBSET=6250,MLP_TAG=mlp_50k,BASELINE_TAG=baseline_moe_6250 run_mlp.slurm
```

This trains the MLP adapter on 50,000 examples and evaluates both the MoE baseline and MLP adapter on 6,250 test examples. The resulting run tags are:

```text
baseline_moe_6250
mlp_50k
```

The generated files follow the run tags:

```text
data/amazon/tst_results_baseline_moe_6250.jsonl
data/amazon/tst_results_mlp_50k.jsonl
```

To run a larger full-train/full-test configuration:

```bash
sbatch --export=ALL,TRAIN_SUBSET=999999,TEST_SUBSET=999999,MLP_TAG=mlp_fulltrain_fulltest,BASELINE_TAG=baseline_moe_fulltest run_mlp.slurm
```

This configuration uses the largest available training and test subsets permitted by the data loader and produces run tags such as:

```text
baseline_moe_fulltest
mlp_fulltrain_fulltest
```

---

## Notes on Experimental Interpretation

This ablation compares a trained MLP adapter against the available released MoE baseline checkpoint. Due to compute and timeline constraints, the original MoE adapter was not retrained under the same training schedule as the MLP. Therefore, the results of this experiment should be read as a comparison between the trained MLP adapter and the available MoE-based XRec baseline, not as a final claim that all MLP variants would underperform.

A more complete future study should retrain both the MoE and MLP adapters under the same training schedule and evaluate multiple random seeds.
