import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "sft_qwen4b_lora.yaml"


def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repo_id = config["model"].get("source_repo")
    local_dir = config["model"].get("model_name_or_path")
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    payload = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "snapshot_path": result,
    }
    out = ROOT / "data" / "reports" / "model_download_report.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
