import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge


ZH_USER = "\u7528\u6237"
ZH_VISITOR = "\u6765\u8bbf\u8005"
ZH_HELP_SEEKER = "\u6c42\u52a9\u8005"
ZH_CONSULTEE = "\u54a8\u8be2\u8005"
ZH_SUPPORTER = "\u652f\u6301\u8005"
ZH_SYSTEM = "\u7cfb\u7edf"
ZH_REPLY = "\u56de\u590d"
ZH_ANSWER = "\u56de\u7b54"
ZH_OK = "\u597d\u7684"
ZH_SURE = "\u5f53\u7136\u53ef\u4ee5"
ZH_PREFIX = "\u4ee5\u4e0b\u662f"
FULLWIDTH_COLON = "\uff1a"

CUT_MARKERS = [
    "<user>",
    "<system>",
    "<assistant>",
    "<ai>",
    f"{ZH_USER}:",
    f"{ZH_USER}{FULLWIDTH_COLON}",
    f"{ZH_VISITOR}:",
    f"{ZH_VISITOR}{FULLWIDTH_COLON}",
    f"{ZH_HELP_SEEKER}:",
    f"{ZH_HELP_SEEKER}{FULLWIDTH_COLON}",
    "system:",
    f"system{FULLWIDTH_COLON}",
    "user:",
    f"user{FULLWIDTH_COLON}",
    "assistant:",
    f"assistant{FULLWIDTH_COLON}",
    "ai:",
    f"ai{FULLWIDTH_COLON}",
]

LEADING_PREFIX_PATTERNS = [
    re.compile(r"^\s*<(system|user|assistant|ai)>\s*", re.IGNORECASE),
    re.compile(r"^\s*(system|user|assistant|ai)\s*[:\uff1a]\s*", re.IGNORECASE),
    re.compile(
        r"^\s*(%s|%s|%s|%s|%s|%s|%s|%s)\s*[:\uff1a]\s*"
        % (
            ZH_USER,
            ZH_VISITOR,
            ZH_HELP_SEEKER,
            ZH_CONSULTEE,
            ZH_SUPPORTER,
            ZH_SYSTEM,
            ZH_REPLY,
            ZH_ANSWER,
        )
    ),
]

LEADING_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*%s[,\uff0c\u3002\uff01!\s]*" % ZH_OK),
    re.compile(r"^\s*%s[,\uff0c\u3002\uff01!\s]*" % ZH_SURE),
    re.compile(r"^\s*%s[^\u3002:\n]*[:\uff1a]\s*" % ZH_PREFIX),
]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text: str):
    return re.sub(r"\s+", " ", str(text).strip())


def strip_leading_prefixes(text: str):
    value = str(text).strip()
    changed = False
    keep_stripping = True
    while keep_stripping and value:
        keep_stripping = False
        for pattern in LEADING_PREFIX_PATTERNS:
            updated = pattern.sub("", value, count=1)
            if updated != value:
                value = updated.strip()
                changed = True
                keep_stripping = True
    return value, changed


def remove_trailing_dialogue(text: str):
    value = str(text).strip()
    positions = []
    lower_value = value.lower()
    for marker in CUT_MARKERS:
        start = 1 if marker.startswith("<") else 0
        pos = lower_value.find(marker.lower(), start)
        if pos != -1:
            positions.append(pos)
    if positions:
        return value[: min(positions)].strip(), True
    return value, False


def remove_leading_boilerplate(text: str):
    value = str(text).strip()
    changed = False
    keep_cleaning = True
    while keep_cleaning and value:
        keep_cleaning = False
        for pattern in LEADING_BOILERPLATE_PATTERNS:
            updated = pattern.sub("", value, count=1)
            if updated != value:
                value = updated.strip()
                changed = True
                keep_cleaning = True
    return value, changed


def clean_generation(text: str):
    value = normalize_text(text)
    value, removed_prefix = strip_leading_prefixes(value)
    value, removed_dialogue = remove_trailing_dialogue(value)
    value, removed_boilerplate = remove_leading_boilerplate(value)
    value = normalize_text(value)
    return value, {
        "cleaned": value != normalize_text(text),
        "removed_role_prefix": removed_prefix,
        "removed_trailing_dialogue": removed_dialogue,
        "removed_leading_boilerplate": removed_boilerplate,
    }


def tokenize_zh(text: str):
    value = normalize_text(text).replace(" ", "")
    if not value:
        return []
    return list(jieba.cut(value))


def compute_metrics(prediction: str, reference: str):
    pred_tokens = tokenize_zh(prediction)
    ref_tokens = tokenize_zh(reference)
    pred_text = " ".join(pred_tokens)
    ref_text = " ".join(ref_tokens)

    if not pred_tokens or not ref_tokens:
        return {
            "bleu-1": 0.0,
            "bleu-2": 0.0,
            "bleu-3": 0.0,
            "bleu-4": 0.0,
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
            "total_score": 0.0,
        }

    smoothing = SmoothingFunction().method1
    bleu_weights = {
        "bleu-1": (1.0, 0.0, 0.0, 0.0),
        "bleu-2": (0.5, 0.5, 0.0, 0.0),
        "bleu-3": (1 / 3, 1 / 3, 1 / 3, 0.0),
        "bleu-4": (0.25, 0.25, 0.25, 0.25),
    }

    scores = {}
    for name, weights in bleu_weights.items():
        scores[name] = sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=smoothing) * 100

    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_text, ref_text, avg=True)
    scores["rouge-1"] = rouge_scores["rouge-1"]["f"] * 100
    scores["rouge-2"] = rouge_scores["rouge-2"]["f"] * 100
    scores["rouge-l"] = rouge_scores["rouge-l"]["f"] * 100

    bleu_score = (scores["bleu-1"] + scores["bleu-2"] + scores["bleu-3"] + scores["bleu-4"]) / 4
    rouge_score = (scores["rouge-1"] + scores["rouge-2"] + scores["rouge-l"]) / 3
    scores["total_score"] = bleu_score * 0.5 + rouge_score * 0.5
    return scores


def resolve_output_path(predictions_path: Path, output_arg: Optional[str]):
    if output_arg:
        return Path(output_arg)
    default_dir = predictions_path.parent / "results"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"{predictions_path.stem}_metrics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSONL file with fields: id, prediction, reference")
    parser.add_argument(
        "--output",
        default=None,
        help="Output json path. Defaults to <predictions_dir>/results/<predictions_stem>_metrics.json",
    )
    parser.add_argument(
        "--disable_cleaning",
        action="store_true",
        help="Disable prediction/reference cleaning before metric computation.",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_path = resolve_output_path(predictions_path, args.output)

    metrics_sum = defaultdict(float)
    count = 0
    details = []

    for item in load_jsonl(predictions_path):
        raw_prediction = item["prediction"]
        raw_reference = item["reference"]

        if args.disable_cleaning:
            prediction = normalize_text(raw_prediction)
            reference = normalize_text(raw_reference)
            cleaning_info = {
                "cleaned": False,
                "removed_role_prefix": False,
                "removed_trailing_dialogue": False,
                "removed_leading_boilerplate": False,
            }
        else:
            prediction, cleaning_info = clean_generation(raw_prediction)
            reference, _ = clean_generation(raw_reference)

        scores = compute_metrics(prediction, reference)
        for key, value in scores.items():
            metrics_sum[key] += value

        details.append(
            {
                "id": item.get("id"),
                "scores": scores,
                "cleaned_prediction": prediction,
                "cleaning": cleaning_info,
            }
        )
        count += 1

    averages = {key: (value / count if count else 0.0) for key, value in metrics_sum.items()}
    payload = {
        "count": count,
        "cleaning_enabled": not args.disable_cleaning,
        "average_metrics": averages,
        "details": details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "average_metrics": averages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
