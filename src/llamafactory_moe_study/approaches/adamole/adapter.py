import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

from llamafactory_moe_study.core.generic_adapter import MethodAdapter
from llamafactory_moe_study.approaches.adamole.adamole.config import AdaMoleConfig
from llamafactory_moe_study.approaches.adamole.utils.peft_types import TaskType
from llamafactory_moe_study.approaches.adamole.peft_model import PeftModelForCausalLM
from llamafactory_moe_study.core.logger import log_adamole_architecture


class AdaMoLEAdapter(MethodAdapter):
    """Adapter wrapper for AdaMoLE"""

    name = "adamole"

    def apply(self, model_args, data_args, training_args):
        # TODO: we first load base model and then use PeftModel loading, differnt than MoLa adapter, is this okay for comparison, does this make a difference?
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args["model_name_or_path"],
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_args["model_name_or_path"],
            padding_side="left",
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        # configure hacked adamole PeftConfig
        peft_config = AdaMoleConfig(
            lora_rank=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_targets.split(","),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            num_experts=training_args.adamole_num_experts,
            max_threshold=training_args.adamole_max_threshold,
        )

        # load hacked admole model
        model = PeftModelForCausalLM(base_model, peft_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        log_adamole_architecture(model)
        return model, tokenizer, {}

    def aux_loss(self, model):
        if hasattr(model, "aux_loss") and callable(model.get_aux_loss):
            aux = model.get_aux_loss()
            if aux is None or not isinstance(aux, torch.Tensor):
                warnings.warn("aux_loss returned None or invalid type, using 0.0 tensor instead.")
                return torch.tensor(0.0, device=next(model.parameters()).device)
            return aux
        warnings.warn("Model has no get_aux_loss, using 0.0 tensor.")
        return torch.tensor(0.0, device=next(model.parameters()).device)

