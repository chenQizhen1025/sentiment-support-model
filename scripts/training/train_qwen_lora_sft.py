import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str, project_root: Path):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def resolve_stage_files(config, stage_name: str):
    stage_cfg = config["data"][stage_name]
    return stage_cfg["train_file"], stage_cfg["dev_file"]


def build_output_dir(config, stage_name: str, project_root: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = resolve_path(config["training"]["output_root"], project_root)
    out_dir = root / stage_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, timestamp


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_texts(tokenizer, prompt_messages, response):
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + response
    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token
    return prompt_text, full_text


def tokenize_example(example, tokenizer, max_length):
    prompt_text, full_text = build_texts(tokenizer, example["prompt_messages"], example["response"])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)

    labels = full_ids.copy()
    for idx in range(min(prompt_len, len(labels))):
        labels[idx] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--stage", required=True, choices=["sft_stage1", "sft_stage2"])
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = load_yaml(config_path)
    set_seed(config["training"].get("seed", 42))

    train_file, dev_file = resolve_stage_files(config, args.stage)
    train_file = str(resolve_path(train_file, project_root))
    dev_file = str(resolve_path(dev_file, project_root))
    output_dir, run_timestamp = build_output_dir(config, args.stage, project_root)
    save_json(output_dir / "resolved_config.json", config)

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_path(config["model"]["model_name_or_path"], project_root)),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        use_fast=config["model"].get("use_fast", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = config["model"].get("torch_dtype", "bfloat16")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        str(resolve_path(config["model"]["model_name_or_path"], project_root)),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        torch_dtype=dtype,
    )

    lora_cfg = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": dev_file,
        },
    )

    max_length = config["data"].get("max_length", 2048)

    def preprocess(example):
        return tokenize_example(example, tokenizer, max_length)

    tokenized = dataset.map(
        preprocess,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing SFT dataset",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=float(config["training"]["num_train_epochs"]),
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        weight_decay=float(config["training"]["weight_decay"]),
        save_total_limit=config["training"]["save_total_limit"],
        bf16=bool(config["training"].get("bf16", False)),
        fp16=bool(config["training"].get("fp16", False)),
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        remove_unused_columns=False,
        report_to=config["training"].get("report_to", "none"),
        seed=config["training"].get("seed", 42),
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 0),
        deepspeed=config["training"].get("deepspeed"),
        ddp_find_unused_parameters=config["training"].get("ddp_find_unused_parameters", False),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    train_metrics = dict(train_result.metrics)
    train_metrics["global_step"] = trainer.state.global_step
    save_json(output_dir / "train_metrics.json", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    save_json(output_dir / "eval_metrics.json", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    save_json(output_dir / "log_history.json", trainer.state.log_history)

    summary = {
        "stage": args.stage,
        "run_timestamp": run_timestamp,
        "config_path": str(config_path),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "train_file": train_file,
        "dev_file": dev_file,
        "model_name_or_path": str(resolve_path(config["model"]["model_name_or_path"], project_root)),
        "train_metrics_file": str(output_dir / "train_metrics.json"),
        "eval_metrics_file": str(output_dir / "eval_metrics.json"),
        "log_history_file": str(output_dir / "log_history.json"),
    }
    save_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
