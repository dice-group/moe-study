import argparse
import yaml
import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from llamafactory_moe_study.approaches.registry import get_selected_method
from llamafactory_moe_study.core.generic_adapter import AdapterConfig
from llamafactory_moe_study.core.wrappers import AuxLossWrapper


def load_yaml_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config file") # this was for llamafactory before, keep it for now TODO:
    parser.add_argument("--use_method", default="adamole")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    parser.add_argument("--adamole_num_experts", type=int, default=4)
    parser.add_argument("--adamole_max_threshold", type=float, default=0.25)
    parser.add_argument("--adamole_aux_coef", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_first_step", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--evaluation_strategy", default="no", choices=["no", "steps", "epoch"])
    args = parser.parse_args()

    # load yaml config
    cfg = load_yaml_config(args.config)

    model_name = cfg["model_name_or_path"]
    dataset_name = cfg["dataset"]
    dataset_dir = cfg["dataset_dir"]
    cutoff_len = cfg.get("cutoff_len", 1024)
    max_samples = cfg.get("max_samples", None)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    # try:
    #     dataset = load_from_disk(dataset_dir)
    # except Exception:
    #     dataset = load_dataset("json", data_files={ "train": f"{dataset_dir}/{dataset_name}.json" })#
    # for testing, TODO:    
    dataset = load_dataset("Yelp/yelp_review_full", split="train[:1000]")
    dataset = dataset.train_test_split(test_size=0.1) 

    # easy tokenizing for now
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cutoff_len)

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    if max_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if cfg.get("fp16", True) else torch.float32,
        device_map="auto"
    )

    # load approach, for now inject Ada Moe adapter
    adapter = get_selected_method(args.use_method)
    knobs = AdapterConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","),
    )
    
    model = adapter.inject(
        model,
        knobs,
        target_modules=args.lora_targets.split(","),
        num_experts=args.adamole_num_experts,
        max_threshold=args.adamole_max_threshold,
    )
    model.to("cuda")

    # wrap aux loss
    def get_aux_loss():
        return adapter.aux_loss(model)
    model.get_aux_loss = get_aux_loss
    model = AuxLossWrapper(model, coef=args.adamole_aux_coef)

    # define training argumetn
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=cfg.get("overwrite_output_dir", True),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=args.gradient_checkpointing,
        logging_first_step=args.logging_first_step,
        report_to=cfg.get("report_to", []),
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
