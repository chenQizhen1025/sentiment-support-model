import json
from pathlib import Path
from statistics import mean

ROOT = Path(r"D:\Sentiment-SUPPORT")
REPORT_DIR = ROOT / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "soulchat_train": ROOT / "data" / "processed" / "train_dedup.jsonl",
    "soulchat_val": ROOT / "data" / "processed" / "val_dedup.jsonl",
    "current_single": ROOT / "data" / "quality_data" / "data_source" / "single_turn_data_18k.json",
    "current_multi_processed": ROOT / "data" / "quality_data" / "data_source" / "multi_turn_data_19k_processed.json",
    "current_multi_raw": ROOT / "data" / "quality_data" / "data_source" / "multi_turn_data_18k.json",
}


def inspect_jsonl_messages(path: Path):
    sample_count = 0
    turn_counts = []
    last_role_counts = {}
    empty_contents = 0
    first_roles = {}
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sample_count += 1
        item = json.loads(line)
        messages = item.get("messages", [])
        turn_counts.append(len(messages))
        if messages:
            first_roles[messages[0].get("role", "unknown")] = first_roles.get(messages[0].get("role", "unknown"), 0) + 1
            last_role = messages[-1].get("role", "unknown")
            last_role_counts[last_role] = last_role_counts.get(last_role, 0) + 1
        for msg in messages:
            if not str(msg.get("content", "")).strip():
                empty_contents += 1
    return {
        "format": "jsonl_messages",
        "sample_count": sample_count,
        "avg_turns": round(mean(turn_counts), 2) if turn_counts else 0,
        "min_turns": min(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
        "first_role_counts": first_roles,
        "last_role_counts": last_role_counts,
        "empty_message_contents": empty_contents,
    }


def inspect_single_turn_json(path: Path):
    data = json.load(path.open("r", encoding="utf-8"))
    empty_question = 0
    empty_answer = 0
    q_lengths = []
    a_lengths = []
    for item in data:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q:
            empty_question += 1
        if not a:
            empty_answer += 1
        q_lengths.append(len(q))
        a_lengths.append(len(a))
    return {
        "format": "single_turn_json",
        "sample_count": len(data),
        "empty_question": empty_question,
        "empty_answer": empty_answer,
        "avg_question_len": round(mean(q_lengths), 2) if q_lengths else 0,
        "avg_answer_len": round(mean(a_lengths), 2) if a_lengths else 0,
        "min_question_len": min(q_lengths) if q_lengths else 0,
        "max_question_len": max(q_lengths) if q_lengths else 0,
        "min_answer_len": min(a_lengths) if a_lengths else 0,
        "max_answer_len": max(a_lengths) if a_lengths else 0,
    }


def inspect_multi_turn_json(path: Path):
    data = json.load(path.open("r", encoding="utf-8"))
    turn_counts = []
    first_roles = {}
    last_roles = {}
    empty_contents = 0
    for item in data:
        messages = item.get("messages", [])
        turn_counts.append(len(messages))
        if messages:
            first_roles[messages[0].get("role", "unknown")] = first_roles.get(messages[0].get("role", "unknown"), 0) + 1
            last_roles[messages[-1].get("role", "unknown")] = last_roles.get(messages[-1].get("role", "unknown"), 0) + 1
        for msg in messages:
            if not str(msg.get("content", "")).strip():
                empty_contents += 1
    return {
        "format": "multi_turn_json",
        "sample_count": len(data),
        "avg_turns": round(mean(turn_counts), 2) if turn_counts else 0,
        "min_turns": min(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
        "first_role_counts": first_roles,
        "last_role_counts": last_roles,
        "empty_message_contents": empty_contents,
    }


def main():
    report = {}
    for name, path in DATASETS.items():
        if not path.exists():
            report[name] = {"exists": False}
            continue
        if path.suffix == ".jsonl":
            info = inspect_jsonl_messages(path)
        else:
            if name == "current_single":
                info = inspect_single_turn_json(path)
            else:
                info = inspect_multi_turn_json(path)
        info["exists"] = True
        info["path"] = str(path)
        report[name] = info

    out_path = REPORT_DIR / "dataset_inventory_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
