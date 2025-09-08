"""
LoRA Model

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
from typing import Any
import warnings

import torch
from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING, ModulesToSaveWrapper
from torch import nn
from transformers import PretrainedConfig

from .config import LoraConfig
from .layer import LoraLayer, LinearLoraLayer


class LoraModel(BaseTuner):
    """
    Low Rank Adapter (LoRA) Model
    """
    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: LoraConfig, adapter_name: str = "default") -> None:
        """
        Initialize LoraModel

        :param model: model to be adapted
        :param config: configuration of the LoRA model
        :param adapter_name: name of the adapter
        """
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str) -> Any:
        """
        Forward missing attributes to the wrapped module
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        Check the config when a new adapter is being added
        """
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. "
                f"When using multiple adapters, set bias to 'none' for all adapters.")

    @staticmethod
    def _check_target_module_exists(lora_config: LoraConfig, key: str) -> bool:
        """
        Check if the passed module's key name matches any of the target modules in the config target module list
        """
        return check_target_module_exists(lora_config, key)

    @staticmethod
    def _prepare_adapter_config(peft_config: LoraConfig, model_config: PretrainedConfig) -> PeftConfig:
        """
        Prepare the adapter config, such as automatically inferring target modules if it is none
        """
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`.")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]],
            )
        return peft_config

    def _create_and_replace(
        self, lora_config: LoraConfig, adapter_name: str,
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
        }

        if isinstance(target, LoraLayer):
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
            new_module = LinearLoraLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module

    def _replace_module(self, parent: nn.Module, child_name: str, new_module: nn.Module, child: nn.Module) -> None:
        """
        Replace the module
        """
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Make only adapters as trainable
        """
        for name, param in model.named_parameters():
            if self.prefix not in name:
                param.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            elif bias == "all":
                for name, param in model.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            elif bias == "lora_only":
                for module in model.modules():
                    if isinstance(module, LoraLayer) and hasattr(module, "bias") and module.bias is not None:
                        module.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters. Call this if you have previously disabled all adapters and want to re-enable them."""
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """
        Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """
        Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...: # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """
        Return the number of trainable parameters and the number of all parameters in the model
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
            f"trainable: {trainable_params / all_param:.2%}")
