# MoE vs. 2-Layer MLP Adapter Ablation for XRec

This documents our ablation study for XRec, which asks the question:
**Can a lightweight 2-layer MLP replace the original XRec Mixture-of-Experts (MoE) adapter without degrading explanation quality?**

The pipeline has five stages:

1. **Baseline Generation** — Run the released pretrained XRec MoE adapter on the test split
2. **MLP Fine-Tuning** — Train a 2-layer MLP adapter while keeping the LLaMA backbone frozen
3. **MLP Generation** — Generate explanations with the trained MLP adapter
4. **Evaluation** — Compare MoE and MLP with standard XRec metrics plus grounding-oriented metrics
5. **Optional Raw BERTScore Check** — Compute raw BERTScore as a sanity check alongside the official rescaled version

---

## Repository Layout

```text
XRec/
├── explainer/
│   ├── main.py                          # Train or generate with MoE / MLP adapters
│   ├── sample.py                        # Preview saved prediction/reference files
│   ├── models/
│   │   ├── explainer.py                 # Adapter definitions + LLaMA wrapper
│   │   └── modeling_explainer.py        # Custom XRec LLaMA backbone
│   └── utils/
│       ├── data_handler.py              # Dataset loading and optional subsampling
│       └── parse.py                     # CLI arguments
│
├── evaluation/
│   ├── main.py                          # Main evaluation entry point
│   ├── metrics.py                       # LlamaScore, BARTScore, BERTScore, ROUGE, BLEU, factual precision
│   ├── compare_results.py               # Side-by-side CSV for two runs
│   └── raw_bertscore.py                 # Optional raw/rescaled BERTScore-only script
│
├── scripts/
│   ├── setup_xrec_hprc_env.sh           # Create TAMU HPRC virtual environment
│   ├── cache_xrec_hprc_models.sh        # Cache LLaMA / BART / BERTScore assets on HPRC
│   ├── run_xrec_mlp_hprc.slurm          # End-to-end HPRC run: MoE baseline + MLP + eval
│   └── run_xrec_eval_only_hprc.slurm    # Evaluation-only HPRC job
│
└── data/
    └── amazon/
        ├── trn.pkl                      # Train split
        ├── val.pkl                      # Validation split
        ├── tst.pkl                      # Test split
        ├── user_emb.pkl                 # LightGCN user embeddings
        ├── item_emb.pkl                 # LightGCN item embeddings
        ├── user_converter.pkl           # Released pretrained MoE user adapter
        ├── item_converter.pkl           # Released pretrained MoE item adapter
        ├── tst_pred_<tag>.pkl           # Generated explanations
        ├── tst_ref_<tag>.pkl            # Ground-truth explanations
        ├── tst_results_<tag>.jsonl      # Structured per-example results
        ├── <tag>_eval.txt               # Saved evaluation output
        └── comparison_<left>_vs_<right>.csv
