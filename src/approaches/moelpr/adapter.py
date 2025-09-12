import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.generic_adapter import MethodAdapter
from approaches.moelpr.model import MoelprModel
from approaches.moelpr.config import MoelprConfig


class MoelprAdapter(MethodAdapter):
    """Adapter wrapper for MoE-LPR"""

    name = "moelpr"

    def apply(self, model_args, data_args, training_args):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args["model_name_or_path"],
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args["model_name_or_path"],
            padding_side="left",
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        moelpr_config = MoelprConfig(
            lora_rank=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_targets.split(","),
            moelpr_num_experts=training_args.moelpr_num_experts,
            moelpr_top_k=training_args.moelpr_top_k,
            moelpr_lpr_loss_coef=training_args.moelpr_lpr_loss_coef,
        )
        model = MoelprModel(base_model, moelpr_config)
        model.print_trainable_parameters()

        return model, tokenizer, {}

    def aux_loss(self, model):
        if hasattr(model, "get_aux_loss") and callable(model.get_aux_loss):
            return model.get_aux_loss()
        warnings.warn("Model has no get_aux_loss, using 0.0 tensor.")
        return torch.tensor(0.0, device=next(model.parameters()).device)
