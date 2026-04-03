import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = "你是一位温和、共情、克制的心理支持助手，请结合来访者的处境给出自然、具体、支持性的回应。"
OUT_DIR = ROOT / "data" / "interim" / "standardized"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_message(msg):
    return {
        "role": str(msg.get("role", "")).strip().lower(),
        "content": str(msg.get("content", "")).strip(),
    }


def with_system(messages):
    return [{"role": "system", "content": SYSTEM_PROMPT}] + messages


def standardize_single_turn():
    src = ROOT / "data" / "quality_data" / "data_source" / "single_turn_data_18k.json"
    dst = OUT_DIR / "current_single_standardized.jsonl"
    data = json.load(src.open("r", encoding="utf-8"))
    with dst.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(data):
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            record = {
                "sample_id": f"current_single_{idx}",
                "source": "current_project_single_turn",
                "topic": item.get("topic", ""),
                "messages": with_system([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dst


def standardize_multi_json(src_path: Path, source_name: str, out_name: str):
    dst = OUT_DIR / out_name
    data = json.load(src_path.open("r", encoding="utf-8"))
    with dst.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(data):
            messages = [normalize_message(m) for m in item.get("messages", [])]
            record = {
                "sample_id": str(item.get("id", f"{source_name}_{idx}")),
                "source": source_name,
                "topic": item.get("topic", ""),
                "messages": with_system(messages),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dst


def standardize_soulchat_jsonl(src_path: Path, source_name: str, out_name: str):
    dst = OUT_DIR / out_name
    with src_path.open("r", encoding="utf-8") as src, dst.open("w", encoding="utf-8") as dst_f:
        for idx, line in enumerate(src):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = [normalize_message(m) for m in item.get("messages", [])]
            record = {
                "sample_id": str(item.get("id", f"{source_name}_{idx}")),
                "source": source_name,
                "topic": item.get("topic", ""),
                "messages": with_system(messages),
            }
            dst_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dst


def main():
    outputs = []
    outputs.append(standardize_single_turn())
    outputs.append(standardize_multi_json(
        ROOT / "data" / "quality_data" / "data_source" / "multi_turn_data_19k_processed.json",
        "current_project_multi_processed",
        "current_multi_processed_standardized.jsonl",
    ))
    outputs.append(standardize_multi_json(
        ROOT / "data" / "quality_data" / "data_source" / "multi_turn_data_18k.json",
        "current_project_multi_raw",
        "current_multi_raw_standardized.jsonl",
    ))
    outputs.append(standardize_soulchat_jsonl(
        ROOT / "data" / "processed" / "train_dedup.jsonl",
        "soulchat_train",
        "soulchat_train_standardized.jsonl",
    ))
    outputs.append(standardize_soulchat_jsonl(
        ROOT / "data" / "processed" / "val_dedup.jsonl",
        "soulchat_val",
        "soulchat_val_standardized.jsonl",
    ))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
