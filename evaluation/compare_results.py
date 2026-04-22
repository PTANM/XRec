import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a side-by-side CSV from two XRec JSONL result files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon",
        help="Dataset name used to resolve default result paths.",
    )
    parser.add_argument(
        "--left_tag",
        type=str,
        default="baseline_moe",
        help="Run tag for the left-side system in the CSV.",
    )
    parser.add_argument(
        "--right_tag",
        type=str,
        default="mlp",
        help="Run tag for the right-side system in the CSV.",
    )
    parser.add_argument(
        "--left_path",
        type=str,
        default="",
        help="Optional explicit JSONL path for the left-side system.",
    )
    parser.add_argument(
        "--right_path",
        type=str,
        default="",
        help="Optional explicit JSONL path for the right-side system.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional explicit output CSV path.",
    )
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_key(record):
    return int(record["uid"]), int(record["iid"])


def pick_text(record, field):
    if record is None:
        return ""
    return str(record.get(field, ""))


def pick_run_field(record, field, fallback=""):
    if record is None:
        return fallback
    value = record.get(field)
    return fallback if value is None else str(value)


def main():
    args = parse_args()

    left_path = Path(args.left_path or f"data/{args.dataset}/tst_results_{args.left_tag}.jsonl")
    right_path = Path(args.right_path or f"data/{args.dataset}/tst_results_{args.right_tag}.jsonl")
    output_path = Path(
        args.output_path
        or f"data/{args.dataset}/comparison_{args.left_tag}_vs_{args.right_tag}.csv"
    )

    left_records = load_jsonl(left_path)
    right_records = load_jsonl(right_path)

    left_map = {to_key(record): record for record in left_records}
    right_map = {to_key(record): record for record in right_records}
    all_keys = sorted(set(left_map) | set(right_map))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "uid",
                "iid",
                "title",
                "item_summary",
                "item_metadata_text",
                "reference",
                f"{args.left_tag}_prediction",
                f"{args.right_tag}_prediction",
                f"{args.left_tag}_adapter_type",
                f"{args.right_tag}_adapter_type",
                f"{args.left_tag}_run_tag",
                f"{args.right_tag}_run_tag",
                f"{args.left_tag}_word_count",
                f"{args.right_tag}_word_count",
                "reference_word_count",
                "present_in_left",
                "present_in_right",
            ],
        )
        writer.writeheader()

        for key in all_keys:
            left_record = left_map.get(key)
            right_record = right_map.get(key)
            canonical = left_record or right_record

            left_prediction = pick_text(left_record, "prediction")
            right_prediction = pick_text(right_record, "prediction")
            reference = pick_text(canonical, "reference")

            writer.writerow(
                {
                    "uid": key[0],
                    "iid": key[1],
                    "title": pick_text(canonical, "title"),
                    "item_summary": pick_text(canonical, "item_summary"),
                    "item_metadata_text": pick_text(canonical, "item_metadata_text"),
                    "reference": reference,
                    f"{args.left_tag}_prediction": left_prediction,
                    f"{args.right_tag}_prediction": right_prediction,
                    f"{args.left_tag}_adapter_type": pick_run_field(
                        left_record, "adapter_type", fallback=args.left_tag
                    ),
                    f"{args.right_tag}_adapter_type": pick_run_field(
                        right_record, "adapter_type", fallback=args.right_tag
                    ),
                    f"{args.left_tag}_run_tag": pick_run_field(
                        left_record, "run_tag", fallback=args.left_tag
                    ),
                    f"{args.right_tag}_run_tag": pick_run_field(
                        right_record, "run_tag", fallback=args.right_tag
                    ),
                    f"{args.left_tag}_word_count": len(left_prediction.split()),
                    f"{args.right_tag}_word_count": len(right_prediction.split()),
                    "reference_word_count": len(reference.split()),
                    "present_in_left": left_record is not None,
                    "present_in_right": right_record is not None,
                }
            )

    print(f"Loaded {len(left_records)} rows from {left_path}")
    print(f"Loaded {len(right_records)} rows from {right_path}")
    print(f"Wrote side-by-side CSV to {output_path}")


if __name__ == "__main__":
    main()
