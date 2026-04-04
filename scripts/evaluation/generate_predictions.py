import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_output_path(eval_path: Path, output_arg: str | None, model_label: str):
    if output_arg:
        return Path(output_arg)
    out_dir = eval_path.parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = model_label.replace("/", "_").replace("\\", "_")
    return out_dir / f"{safe_label}_predictions.jsonl"


def get_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(name, torch.float16)


def build_model(base_model_path: str, adapter_path: str | None, torch_dtype: str, trust_remote_code: bool):
    dtype = get_dtype(torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True, help="JSONL eval file with fields: id, messages, reference")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--model_label", default=None, help="Label used in output filename and records")
    parser.add_argument("--output", default=None, help="Output predictions jsonl path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--torch_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_fast", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    model_label = args.model_label or (Path(args.adapter_path).name if args.adapter_path else Path(args.base_model).name)
    output_path = resolve_output_path(eval_path, args.output, model_label)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args.base_model, args.adapter_path, args.torch_dtype, args.trust_remote_code)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = list(load_jsonl(eval_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": args.do_sample,
            }
            if args.do_sample:
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                outputs = model.generate(**inputs, **generate_kwargs)

            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for item, output_ids, input_len in zip(batch, outputs, input_lengths):
                generated_ids = output_ids[int(input_len):]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                row = {
                    "id": item.get("id"),
                    "model": model_label,
                    "type": item.get("type"),
                    "category": item.get("category"),
                    "prediction": prediction,
                    "reference": item.get("reference", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output_path": str(output_path), "count": len(records), "model": model_label}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
