# methods/base.py
from dataclasses import dataclass
from typing import Iterable
import torch.nn as nn

@dataclass
class AdapterConfig:
    """
    Generic adapter configuration which holds the tunable hyperparameters of each approach
    """
    r: int
    alpha: int
    dropout: float
    target_modules: Iterable[str]

class MethodAdapter:
    """Interface that each adapter method must implement."""
    name: str = "BASE"

    def inject(self, model: nn.Module, cfg: AdapterConfig) -> nn.Module:
        """Replace target modules in model using given adapter config."""
        raise NotImplementedError

    def aux_loss(self, model: nn.Module):
        """Return method-specific auxiliary loss if any."""
        raise NotImplementedError
