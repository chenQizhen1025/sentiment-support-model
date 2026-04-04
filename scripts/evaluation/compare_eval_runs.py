import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

METRIC_ORDER = [
    "bleu-1",
    "bleu-2",
    "bleu-3",
    "bleu-4",
    "rouge-1",
    "rouge-2",
    "rouge-l",
    "total_score",
]


def parse_input_item(raw: str):
    if "=" not in raw:
        raise ValueError(f"Invalid input '{raw}'. Expected format: label=path/to/predictions.jsonl")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise ValueError(f"Invalid input '{raw}'. Label cannot be empty.")
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    return label, path


def resolve_output_dir(inputs, output_arg: Optional[str]):
    if output_arg:
        output_dir = Path(output_arg)
    else:
        output_dir = inputs[0][1].parent / "comparison_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_auto_eval(script_path: Path, predictions_path: Path, output_path: Path, disable_cleaning: bool):
    cmd = [
        sys.executable,
        str(script_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(output_path),
    ]
    if disable_cleaning:
        cmd.append("--disable_cleaning")
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def load_metrics(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "count"] + METRIC_ORDER)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_table(rows):
    header = ["model", "count"] + METRIC_ORDER
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [str(row["model"]), str(row["count"])] + [f"{row[key]:.4f}" for key in METRIC_ORDER]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of label=predictions.jsonl pairs, for example: base=base.jsonl stage1=stage1.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for per-model metrics and comparison summary. Defaults to <first_predictions_dir>/comparison_results",
    )
    parser.add_argument(
        "--disable_cleaning",
        action="store_true",
        help="Disable cleaning when invoking auto_eval.py.",
    )
    args = parser.parse_args()

    inputs = [parse_input_item(item) for item in args.inputs]
    output_dir = resolve_output_dir(inputs, args.output_dir)
    auto_eval_script = Path(__file__).with_name("auto_eval.py")

    comparison_rows = []
    comparison_payload = {
        "output_dir": str(output_dir),
        "cleaning_enabled": not args.disable_cleaning,
        "models": {},
    }

    for label, predictions_path in inputs:
        metrics_path = output_dir / f"{label}_metrics.json"
        stdout = run_auto_eval(auto_eval_script, predictions_path, metrics_path, args.disable_cleaning)
        payload = load_metrics(metrics_path)
        averages = payload.get("average_metrics", {})

        row = {"model": label, "count": payload.get("count", 0)}
        for metric in METRIC_ORDER:
            row[metric] = float(averages.get(metric, 0.0))
        comparison_rows.append(row)

        comparison_payload["models"][label] = {
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "count": payload.get("count", 0),
            "average_metrics": averages,
            "auto_eval_stdout": stdout,
        }

    comparison_rows.sort(key=lambda row: row["total_score"], reverse=True)
    comparison_payload["ranking"] = comparison_rows
    comparison_payload["best_model"] = comparison_rows[0]["model"] if comparison_rows else None

    summary_json = output_dir / "comparison_summary.json"
    summary_csv = output_dir / "comparison_table.csv"
    summary_md = output_dir / "comparison_table.md"

    summary_json.write_text(json.dumps(comparison_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(summary_csv, comparison_rows)
    summary_md.write_text(build_markdown_table(comparison_rows), encoding="utf-8")

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "best_model": comparison_payload["best_model"],
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
