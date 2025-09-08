import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

from llamafactory_moe_study.core.generic_adapter import MethodAdapter
from llamafactory_moe_study.approaches.mola.model import MolaModel
from llamafactory_moe_study.approaches.mola.config import MolaConfig


class MolaAdapter(MethodAdapter):
    """Adapter wrapper for MoLA"""

    name = "mola"

    def apply(self, model_args, data_args, training_args):
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args["model_name_or_path"],
            torch_dtype=torch.float16,
        )

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args["model_name_or_path"],
            padding_side="left",
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Load MoLA model
        mola_config = MolaConfig(
            lora_rank=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_targets.split(","),
            mola_num_experts=training_args.mola_num_experts,
            mola_top_k=training_args.mola_top_k,
        )
        model = MolaModel(base_model, mola_config)
        model.print_trainable_parameters()

        return model, tokenizer, {}

    def aux_loss(self, model):
        if hasattr(model, "get_aux_loss") and callable(model.get_aux_loss):
            return model.get_aux_loss()
        warnings.warn("Model has no get_aux_loss, using 0.0 tensor.")
        return torch.tensor(0.0, device=next(model.parameters()).device)
