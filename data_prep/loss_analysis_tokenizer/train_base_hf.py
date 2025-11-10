import argparse
import importlib.util
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from lm_eval import LMEvalCallback

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, default=None)
parser.add_argument("--run_name", type=str, default="llama32-run")
parser.add_argument("--per_device_batch_size", type=int, default=4)
parser.add_argument("--grad_accum_steps", type=int, default=2)
parser.add_argument("--enable_lm_eval", action="store_true", help="Enable lm-harness evaluation callback.")
parser.add_argument("--lm_eval_interval", type=int, default=2000, help="Steps between lm-harness evaluations (match save_steps).")
parser.add_argument(
    "--lm_eval_tasks",
    nargs="+",
    default=["belebele_lao_Laoo", "belebele_ory_Orya", "belebele_kat_Geor", "belebele_tel_Telu"],
    help="Tasks to evaluate with lm-harness.",
)
parser.add_argument("--lm_eval_batch_size", type=int, default=2, help="Per-device batch size for lm-eval CLI.")
parser.add_argument("--lm_eval_limit", type=int, default=500, help="Maximum samples per lm-eval task.")
parser.add_argument("--lm_eval_cuda_devices", type=str, default="0", help="CUDA devices string for lm-eval subprocess.")
parser.add_argument("--lm_eval_wandb_project", type=str, default="lm_eval_project", help="W&B project for lm-eval logging.")
parser.add_argument("--lm_eval_wandb_run_id", type=str, default=None, help="Optional W&B run ID to resume lm-eval logging.")
parser.add_argument(
    "--lm_eval_model_path",
    type=str,
    default=None,
    help="Base model path passed to lm-eval (defaults to --model_name).",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if len(tokenizer) != model.config.vocab_size:
    print(f"Resize embeddings from {model.config.vocab_size}  to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("json", data_files={"train": args.dataset_path})

def tokenize_fn(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=4096,
        padding="max_length",
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = dataset.map(tokenize_fn, batched=True, num_proc=16, remove_columns=dataset["train"].column_names)
eval_dataset = tokenized["validation"] if "validation" in tokenized else None
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    run_name=args.run_name,
    report_to=["wandb"],
    per_device_train_batch_size=args.per_device_batch_size,
    gradient_accumulation_steps=args.grad_accum_steps,
    learning_rate=2e-5,
    max_steps=10000,
    weight_decay=0.1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=True,
    gradient_checkpointing=False,
    dataloader_pin_memory=True,
    logging_steps=25,
    save_steps=2000,
    save_total_limit=20,
)

callbacks = []
if args.enable_lm_eval:
    callbacks.append(
        LMEvalCallback(
            model_path=args.lm_eval_model_path or args.model_name,
            eval_interval=args.lm_eval_interval,
            eval_tasks=args.lm_eval_tasks,
            batch_size=args.lm_eval_batch_size,
            limit=args.lm_eval_limit,
            cuda_devices=args.lm_eval_cuda_devices,
            wandb_project=args.lm_eval_wandb_project,
            wandb_run_id=args.lm_eval_wandb_run_id,
        )
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=callbacks,
)

trainer.train()
