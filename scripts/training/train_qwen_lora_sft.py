import argparse
import inspect
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
    EarlyStoppingCallback,
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


def build_training_arguments(config, output_dir: Path):
    training_cfg = config["training"]
    signature = inspect.signature(TrainingArguments.__init__)
    supported = signature.parameters

    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": training_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "learning_rate": float(training_cfg["learning_rate"]),
        "num_train_epochs": float(training_cfg["num_train_epochs"]),
        "logging_steps": training_cfg["logging_steps"],
        "eval_steps": training_cfg["eval_steps"],
        "save_steps": training_cfg["save_steps"],
        "lr_scheduler_type": training_cfg["lr_scheduler_type"],
        "weight_decay": float(training_cfg["weight_decay"]),
        "save_total_limit": training_cfg.get("save_total_limit", 1),
        "bf16": bool(training_cfg.get("bf16", False)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "remove_unused_columns": False,
        "report_to": training_cfg.get("report_to", "none"),
        "seed": training_cfg.get("seed", 42),
        "dataloader_num_workers": training_cfg.get("dataloader_num_workers", 0),
        "deepspeed": training_cfg.get("deepspeed"),
        "ddp_find_unused_parameters": training_cfg.get("ddp_find_unused_parameters", False),
        "load_best_model_at_end": bool(training_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(training_cfg.get("greater_is_better", False)),
        "group_by_length": bool(training_cfg.get("group_by_length", False)),
    }

    if "warmup_steps" in training_cfg:
        kwargs["warmup_steps"] = int(training_cfg["warmup_steps"])
    else:
        kwargs["warmup_ratio"] = float(training_cfg.get("warmup_ratio", 0.0))

    if "length_column_name" in supported and training_cfg.get("length_column_name"):
        kwargs["length_column_name"] = training_cfg["length_column_name"]

    if "dataloader_pin_memory" in supported:
        kwargs["dataloader_pin_memory"] = bool(training_cfg.get("dataloader_pin_memory", True))

    if "dataloader_persistent_workers" in supported:
        kwargs["dataloader_persistent_workers"] = bool(
            training_cfg.get("dataloader_persistent_workers", training_cfg.get("dataloader_num_workers", 0) > 0)
        )

    if "eval_accumulation_steps" in supported and training_cfg.get("eval_accumulation_steps") is not None:
        kwargs["eval_accumulation_steps"] = int(training_cfg["eval_accumulation_steps"])

    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if "save_only_model" in supported:
        kwargs["save_only_model"] = bool(training_cfg.get("save_only_model", True))

    return TrainingArguments(**kwargs)



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
        "length": len(full_ids),
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

    model_kwargs = {
        "trust_remote_code": config["model"].get("trust_remote_code", True),
    }
    if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        str(resolve_path(config["model"]["model_name_or_path"], project_root)),
        **model_kwargs,
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

    training_args = build_training_arguments(config, output_dir)

    callbacks = []
    early_stopping_patience = config["training"].get("early_stopping_patience")
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=float(config["training"].get("early_stopping_threshold", 0.0)),
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=callbacks,
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    train_metrics = dict(train_result.metrics)
    train_metrics["global_step"] = trainer.state.global_step
    train_metrics["best_metric"] = trainer.state.best_metric
    train_metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
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
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics_file": str(output_dir / "train_metrics.json"),
        "eval_metrics_file": str(output_dir / "eval_metrics.json"),
        "log_history_file": str(output_dir / "log_history.json"),
    }
    save_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

