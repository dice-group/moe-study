# approaches/mola/adapter.py

import torch
from transformers import AutoConfig, LlamaTokenizer
from llamafactory_moe_study.core.generic_adapter import MethodAdapter

from llamafactory_moe_study.approaches.mola.mola_modeling_llama_hacked import LlamaForCausalLM_d
from llamafactory_moe_study.approaches.mola.mola_mapping_hacked import get_peft_model
from llamafactory_moe_study.approaches.mola.mola_lora_hacked import LoraConfig


class MoLAAdapter(MethodAdapter):
    """Adapter wrapper for MoLA hacked logic"""

    name = "mola"

    def apply(self, model_args, data_args, training_args):
        # Load hacked llama model
        config = AutoConfig.from_pretrained(model_args["model_name_or_path"])
        model = LlamaForCausalLM_d.from_pretrained(
            model_args["model_name_or_path"],
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_args["model_name_or_path"])
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        # mola approach loraconfig
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_targets.split(","),
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # load patched AdaMoE mola model
        model = get_peft_model(
            model,
            lora_config,
            number_experts=[training_args.mola_num_experts] * len(training_args.lora_targets.split(",")),
            top_k=[training_args.mola_top_k] * len(training_args.lora_targets.split(",")),
        )

        return model, tokenizer, {}
    
    def aux_loss(self, model):
        # not necessary, TODO: remove in parent MethodAdapter?
        return torch.tensor(0.0, device=next(model.parameters()).device)