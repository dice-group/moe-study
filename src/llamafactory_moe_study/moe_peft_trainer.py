"""
Trainer
"""
import os
import torch

from typing import Optional, Any, Union
from torch import nn
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME, logger

from llamafactory_moe_study.core.logger import log


class PeftTrainer(Trainer):
    """
    Trainer for the PEFT Model
    """

    def __init__(self, aux_loss_coeff=1e-2, **kwargs):
        """
        Initialize PeftTrainer

        :param aux_loss_coeff: a coefficient for the load balancing loss in Mixture-of-Experts (MoE) models
        :param kwargs: additional keyword arguments
        """
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.aux_loss_coeff = aux_loss_coeff

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save the model and tokenizer
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            return_outputs: bool = False,
            num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        Compute loss with optional auxiliary MoE loss.
        """
        outputs = model(**inputs)
        loss = outputs.get("loss")

        if loss is None:
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        # add aux loss for adamole
        if getattr(self, "aux_loss_coeff", None) and hasattr(model, "get_aux_loss"):
            aux = model.get_aux_loss()
            if isinstance(aux, torch.Tensor):
                loss = loss + self.aux_loss_coeff * aux
                log.debug(f"step={self.state.global_step} loss={loss.item():.4f} aux={aux.item():.6f}")
        if (
            self.args.average_tokens_across_devices
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

