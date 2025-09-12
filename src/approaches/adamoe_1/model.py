"""
MoLA Model
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from .config import MolaConfig
from .layer import MolaLayer, LinearMolaLayer
from approaches.adamole.lora import LoraModel


class MolaModel(LoraModel):
    """
    MoLA (Mixture of LoRA Experts) Model
    """
    prefix: str = "lora_"

    def __init__(self, model, config: MolaConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self, lora_config: MolaConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        kwargs = {
            "lora_rank": lora_config.lora_rank,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "init_lora_weights": lora_config.init_lora_weights,
            "num_experts": lora_config.mola_num_experts,
            "top_k": lora_config.mola_top_k,
            "use_null_expert": lora_config.mola_use_null_expert,
            "router_aux_loss_coef": lora_config.mola_router_aux_loss_coef,
            "null_expert_penalty": lora_config.mola_null_expert_penalty,
            "aux_loss_annealing": lora_config.mola_aux_loss_annealing,
            "mola_debug_mode": lora_config.mola_debug_mode,
        }

        if isinstance(target, MolaLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(adapter_name: str, target: nn.Module, **kwargs: Any) -> nn.Module:
        """
        Create the new LoRA module for the target module
        """
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = LinearMolaLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module {target} is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module

    def get_aux_loss(self, adapter_name="default") -> torch.Tensor:
        """
        Get the load balancing loss for the whole model
        """
        model_loss = torch.tensor(0, dtype=torch.float).to(self.model.device)

        for name, module in self.model.named_modules():
            if name.endswith('moe_layer'):
                layer_loss = module[adapter_name].layer_loss
                model_loss += layer_loss

        return model_loss