import argparse
import yaml
import wandb
import torch
import numpy as np
import random

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from approaches.registry import get_selected_method
from core.wrappers import AuxLossWrapper
from core.logger import get_logger
from evaluation import HarnessTrainer

log = get_logger("moelpr")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    
    # mola
    parser.add_argument("--mola_num_experts", type=int, default=4)
    parser.add_argument("--mola_top_k", type=int, default=2)
    parser.add_argument("--mola_use_null_expert", action="store_true")
    parser.add_argument("--mola_router_aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--mola_null_expert_penalty", type=float, default=0.1)
    parser.add_argument("--mola_aux_loss_annealing", action="store_true")
    parser.add_argument("--mola_debug_mode", action="store_true")

    # moelpr
    parser.add_argument("--moelpr_num_experts", type=int, default=4)
    parser.add_argument("--moelpr_top_k", type=int, default=2)
    parser.add_argument("--moelpr_aux_loss_coef", type=float, default=0.01)
    parser.add_argument("--moelpr_lpr_loss_coef", type=float, default=0.01)
    parser.add_argument("--moelpr_layers_to_transform", type=str, default=None, help="layer ids to upcycle to MoE")
    parser.add_argument("--moelpr_stage", type=int, default=1, choices=[1, 2], help="MoE-LPR training stage")
    parser.add_argument("--moelpr_debug_mode", action="store_true")

    parser.add_argument("--tokenized_dataset_dir", required=True, help="path to pretokenized dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_first_step", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    
    parser.add_argument("--evaluation_strategy", default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--do_harness_eval", action="store_true")
    parser.add_argument("--harness_eval_steps", type=int, default=None)
    parser.add_argument("--harness_tasks", nargs="+", default=["belebele"])
    parser.add_argument("--wandb_project", type=str, default="moe-study")
    parser.add_argument("--track_flops", action="store_true", help="Enable FLOPs tracking.")
    
    args = parser.parse_args()
    set_seed(args.seed)

    # load yaml config
    cfg = load_yaml_config(args.config)
    wandb.init(project=args.wandb_project)

    model_name = cfg["model_name_or_path"]
    dataset_name = cfg["dataset"]
    dataset_dir = cfg["dataset_dir"]
    cutoff_len = cfg.get("cutoff_len", 1024)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    if args.use_method == "moelpr":
        log.info(f"[moelpr] Running MoE-LPR in Stage {args.moelpr_stage}")

    log.info(f"Loading tokenized dataset from {args.tokenized_dataset_dir}")
    dataset = load_from_disk(args.tokenized_dataset_dir)
    # test
    dataset = dataset.train_test_split(test_size=0.01, seed=args.seed)

    # load approach, inject specific adapter with specialised MoE approach logic
    adapter = get_selected_method(args.use_method)
    model, tokenizer, _ = adapter.apply(
        cfg,
        {"dataset": dataset_name, "dataset_dir": dataset_dir, "cutoff_len": cutoff_len},
        args,
    )
    model.to("cuda")

    aux_loss_coef = 0.0
    if args.use_method == "adamole":
        aux_loss_coef = args.adamole_aux_coef
    elif args.use_method == "mola":
        aux_loss_coef = args.mola_router_aux_loss_coef
    elif args.use_method == "moelpr":
        aux_loss_coef = args.moelpr_aux_loss_coef
    model = AuxLossWrapper(model, coef=aux_loss_coef)

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
        report_to="wandb",
        remove_unused_columns=False,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        do_eval=args.do_harness_eval,
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer = HarnessTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        harness_tasks=args.harness_tasks,
        do_harness_eval=args.do_harness_eval,
    )
    trainer.train()

    if args.use_method == "moelpr" and args.moelpr_stage == 1:
        log.info("[moelpr] save full model of stage 1")
        model.model.save_pretrained(cfg["output_dir"], safe_serialization=False)
    else:
        model.model.save_pretrained(cfg["output_dir"], safe_serialization=False)

    tokenizer.save_pretrained(cfg["output_dir"])
    model.model.config.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
