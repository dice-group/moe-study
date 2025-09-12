"""
adamole Configuration
"""
from dataclasses import dataclass, field

from approaches.adamole.lora import LoraConfig
from approaches.adamole.utils.peft_types import PeftType


@dataclass
class AdaMoleConfig(LoraConfig):
    """
    adamole Configuration
    """
    num_experts: int = field(default=4, metadata={"help": "The number of experts in MoE."})
    max_threshold: float = field(default=None, metadata={
        "help": "The maximum threshold for selecting experts in the threshold function. "
                "The default value will be 1 / number of experts"})

    def __post_init__(self):
        self.peft_type = PeftType.ADAMOLE
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )