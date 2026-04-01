import json
from pathlib import Path

ROOT = Path(r"D:\Sentiment-SUPPORT")
IN_DIR = ROOT / "data" / "interim" / "standardized"
OUT_DIR = ROOT / "data" / "interim" / "filtered"
REPORT_DIR = ROOT / "data" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ASSISTANT_LEN = 6
MAX_MESSAGE_LEN = 4000


def valid_structure(messages):
    if not messages or len(messages) < 3:
        return False, "too_few_messages"
    if messages[0].get("role") != "system":
        return False, "missing_system"
    roles = [m.get("role") for m in messages[1:]]
    if roles[0] != "user":
        return False, "first_turn_not_user"
    for msg in messages:
        if not str(msg.get("content", "")).strip():
            return False, "empty_content"
        if len(str(msg.get("content", ""))) > MAX_MESSAGE_LEN:
            return False, "message_too_long"
    return True, "ok"


def filter_file(src_path: Path):
    dst_path = OUT_DIR / src_path.name.replace("_standardized", "_filtered")
    kept = 0
    dropped = {}
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            ok, reason = valid_structure(messages)
            if not ok:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            if not assistant_messages:
                dropped["no_assistant"] = dropped.get("no_assistant", 0) + 1
                continue
            if any(len(m.get("content", "").strip()) < MIN_ASSISTANT_LEN for m in assistant_messages):
                dropped["assistant_too_short"] = dropped.get("assistant_too_short", 0) + 1
                continue
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    return {"source": src_path.name, "kept": kept, "dropped": dropped, "output": str(dst_path)}


def main():
    report = []
    for src_path in sorted(IN_DIR.glob("*.jsonl")):
        report.append(filter_file(src_path))
    out_path = REPORT_DIR / "basic_filter_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
