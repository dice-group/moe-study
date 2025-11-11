import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, default=None)
parser.add_argument("--run_name", type=str, default="llama32-run")
parser.add_argument("--per_device_batch_size", type=int, default=4)
parser.add_argument("--grad_accum_steps", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps.")
parser.add_argument("--save_steps", type=int, default=2000, help="Checkpoint save interval.")
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
    max_steps=args.max_steps,
    weight_decay=0.1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    gradient_checkpointing=False,
    dataloader_pin_memory=True,
    logging_steps=25,
    save_steps=args.save_steps,
    save_total_limit=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
