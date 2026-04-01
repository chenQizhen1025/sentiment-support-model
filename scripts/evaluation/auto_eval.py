import argparse
import json
from collections import defaultdict
from pathlib import Path

import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize_zh(text: str):
    return list(jieba.cut(str(text).strip().replace(" ", "")))


def compute_metrics(prediction: str, reference: str):
    pred_tokens = tokenize_zh(prediction)
    ref_tokens = tokenize_zh(reference)
    pred_text = " ".join(pred_tokens)
    ref_text = " ".join(ref_tokens)

    smoothing = SmoothingFunction().method1
    bleu_weights = {
        "bleu-1": (1.0, 0.0, 0.0, 0.0),
        "bleu-2": (0.5, 0.5, 0.0, 0.0),
        "bleu-3": (1/3, 1/3, 1/3, 0.0),
        "bleu-4": (0.25, 0.25, 0.25, 0.25),
    }

    scores = {}
    for name, weights in bleu_weights.items():
        scores[name] = sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=smoothing) * 100

    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_text, ref_text, avg=True)
    scores['rouge-1'] = rouge_scores['rouge-1']['f'] * 100
    scores['rouge-2'] = rouge_scores['rouge-2']['f'] * 100
    scores['rouge-l'] = rouge_scores['rouge-l']['f'] * 100
    scores['total_score'] = (
        scores['bleu-1'] + scores['bleu-2'] + scores['bleu-3'] + scores['bleu-4'] +
        scores['rouge-1'] + scores['rouge-2'] + scores['rouge-l']
    ) / 7
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='JSONL file with fields: id, prediction, reference')
    parser.add_argument('--output', required=True, help='Output json path')
    args = parser.parse_args()

    metrics_sum = defaultdict(float)
    count = 0
    details = []

    for item in load_jsonl(Path(args.predictions)):
        prediction = item['prediction']
        reference = item['reference']
        scores = compute_metrics(prediction, reference)
        for key, value in scores.items():
            metrics_sum[key] += value
        details.append({
            'id': item.get('id'),
            'scores': scores,
        })
        count += 1

    averages = {key: (value / count if count else 0.0) for key, value in metrics_sum.items()}
    payload = {
        'count': count,
        'average_metrics': averages,
        'details': details,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(averages, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
