from dataclasses import dataclass, field
from llamafactory_moe_study.approaches.adamole.lora.config import LoraConfig

@dataclass
class MolaConfig(LoraConfig):
    mola_num_experts: int = field(default=4, metadata={"help": "Number of experts in MoLA"})
    mola_top_k: int = field(default=2, metadata={"help": "Number of experts to use for each token"})
