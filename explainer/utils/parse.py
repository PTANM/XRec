import argparse

def parse_configure():
    parser = argparse.ArgumentParser(description="explainer")
    parser.add_argument("--dataset", type=str, default="amazon", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--mode", type=str, default="finetune", help="finetune or generate")
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="moe",
        choices=["moe", "mlp"],
        help="Adapter architecture to use for the collaborative embeddings.",
    )
    parser.add_argument(
        "--adapter_hidden_size",
        type=int,
        default=512,
        help="Hidden size of the 2-layer MLP adapter.",
    )
    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=0.2,
        help="Dropout applied inside the adapter.",
    )
    parser.add_argument(
        "--checkpoint_tag",
        type=str,
        default="",
        help="Suffix used for run-specific adapter checkpoints and outputs.",
    )
    parser.add_argument(
        "--load_pretrained_adapter",
        action="store_true",
        help="Load the official pretrained MoE adapter checkpoints before generation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Hugging Face model name used for explanation generation.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the frozen LLaMA backbone in 8-bit mode when CUDA is available.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=None,
        help="Optional number of training examples to keep.",
    )
    parser.add_argument(
        "--val_subset_size",
        type=int,
        default=None,
        help="Optional number of validation examples to keep.",
    )
    parser.add_argument(
        "--test_subset_size",
        type=int,
        default=None,
        help="Optional number of test examples to keep.",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="Random seed used for deterministic subsampling.",
    )
    return parser.parse_args()

args = parse_configure()
