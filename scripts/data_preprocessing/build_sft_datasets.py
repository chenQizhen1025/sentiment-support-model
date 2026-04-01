import hashlib
import json
import random
from pathlib import Path

ROOT = Path(r"D:\Sentiment-SUPPORT")
FILTERED_DIR = ROOT / "data" / "interim" / "filtered"
REPORT_DIR = ROOT / "data" / "reports"
STAGE1_DIR = ROOT / "data" / "processed" / "sft_stage1"
STAGE2_DIR = ROOT / "data" / "processed" / "sft_stage2"

SEED = 42
DEV_RATIO = 0.05
TEST_RATIO = 0.05
MAX_HISTORY_MESSAGES = 6
MIN_RESPONSE_LEN = 6

for path in [REPORT_DIR, STAGE1_DIR, STAGE2_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def trim_prompt_messages(messages):
    system_message = messages[0]
    history = messages[1:]
    history = history[-MAX_HISTORY_MESSAGES:]
    while history and history[0].get("role") == "assistant":
        history = history[1:]
    if not history:
        return [system_message]
    return [system_message] + history


def build_single_example(record):
    messages = record.get("messages", [])
    if len(messages) < 3:
        return []
    if messages[-1].get("role") != "assistant":
        return []
    response = normalize_text(messages[-1].get("content", ""))
    if len(response) < MIN_RESPONSE_LEN:
        return []
    return [{
        "sample_id": f"{record.get('sample_id', 'single')}_sft",
        "source": record.get("source", "unknown"),
        "topic": record.get("topic", ""),
        "prompt_messages": messages[:-1],
        "response": response,
    }]


def build_multi_examples(record):
    messages = record.get("messages", [])
    if len(messages) < 3:
        return []
    examples = []
    for idx, message in enumerate(messages):
        if idx == 0:
            continue
        if message.get("role") != "assistant":
            continue
        response = normalize_text(message.get("content", ""))
        if len(response) < MIN_RESPONSE_LEN:
            continue
        prompt_messages = trim_prompt_messages(messages[:idx])
        if len(prompt_messages) < 2:
            continue
        if prompt_messages[1].get("role") != "user":
            continue
        examples.append({
            "sample_id": f"{record.get('sample_id', 'multi')}_turn_{idx}",
            "source": record.get("source", "unknown"),
            "topic": record.get("topic", ""),
            "prompt_messages": prompt_messages,
            "response": response,
        })
    return examples


def deduplicate_examples(examples):
    unique = []
    seen = set()
    duplicates = 0
    for item in examples:
        prompt_text = "\n".join(
            f"{msg.get('role', '')}:{normalize_text(msg.get('content', ''))}"
            for msg in item.get("prompt_messages", [])
        )
        response = normalize_text(item.get("response", ""))
        signature = hashlib.sha1(f"{prompt_text}\n<assistant>{response}".encode("utf-8")).hexdigest()
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature)
        item["dedup_key"] = signature
        unique.append(item)
    return unique, duplicates


def split_examples(examples, seed):
    rng = random.Random(seed)
    items = list(examples)
    rng.shuffle(items)
    total = len(items)
    dev_count = int(total * DEV_RATIO)
    test_count = int(total * TEST_RATIO)
    train_count = total - dev_count - test_count
    return {
        "train": items[:train_count],
        "dev": items[train_count:train_count + dev_count],
        "test": items[train_count + dev_count:],
    }


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = dict(row)
            payload.pop("dedup_key", None)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_source_examples(file_name: str, mode: str):
    src_path = FILTERED_DIR / file_name
    if not src_path.exists():
        raise FileNotFoundError(f"Missing filtered dataset: {src_path}")
    built = []
    input_records = 0
    for record in load_jsonl(src_path):
        input_records += 1
        if mode == "single":
            built.extend(build_single_example(record))
        else:
            built.extend(build_multi_examples(record))
    deduped, duplicate_count = deduplicate_examples(built)
    return {
        "file_name": file_name,
        "mode": mode,
        "input_records": input_records,
        "examples_before_dedup": len(built),
        "examples_after_dedup": len(deduped),
        "duplicates_removed": duplicate_count,
        "examples": deduped,
    }


def build_stage(stage_name: str, sources, out_dir: Path):
    report = {"stage": stage_name, "seed": SEED, "sources": [], "splits": {"train": [], "dev": [], "test": []}}
    for offset, source in enumerate(sources):
        result = build_source_examples(source["file_name"], source["mode"])
        source_examples = result.pop("examples")
        split = split_examples(source_examples, SEED + offset)
        report["sources"].append(result)
        for split_name, rows in split.items():
            report["splits"][split_name].extend(rows)
    split_stats = {}
    for split_name, rows in report["splits"].items():
        write_jsonl(out_dir / f"{split_name}.jsonl", rows)
        split_stats[split_name] = len(rows)
    report_path = REPORT_DIR / f"{stage_name}_dataset_report.json"
    payload = {
        "stage": stage_name,
        "seed": SEED,
        "source_reports": report["sources"],
        "split_sizes": split_stats,
        "output_dir": str(out_dir),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    stage1_sources = [
        {"file_name": "current_single_filtered.jsonl", "mode": "single"},
        {"file_name": "current_multi_processed_filtered.jsonl", "mode": "multi"},
    ]
    stage2_sources = [
        {"file_name": "current_single_filtered.jsonl", "mode": "single"},
        {"file_name": "current_multi_processed_filtered.jsonl", "mode": "multi"},
        {"file_name": "current_multi_raw_filtered.jsonl", "mode": "multi"},
        {"file_name": "soulchat_train_filtered.jsonl", "mode": "multi"},
        {"file_name": "soulchat_val_filtered.jsonl", "mode": "multi"},
    ]
    build_stage("sft_stage1", stage1_sources, STAGE1_DIR)
    build_stage("sft_stage2", stage2_sources, STAGE2_DIR)


if __name__ == "__main__":
    main()
