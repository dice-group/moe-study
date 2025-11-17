from dataclasses import dataclass, field
from approaches.adamole.lora.config import LoraConfig

@dataclass
class MolaConfig(LoraConfig):
    mola_num_experts: int = field(default=4, metadata={"help": "Number of experts in MoLA"})
    mola_top_k: int = field(default=2, metadata={"help": "Number of experts to use for each token"})
    mola_use_null_expert: bool = field(default=False, metadata={"help": "Whether to use a null expert in MoLA"})
    mola_router_aux_loss_coef: float = field(default=0.01, metadata={"help": "The coefficient for the load balancing loss."})
    mola_null_expert_penalty: float = field(default=0.1, metadata={"help": "The penalty for not using the null experts."})
    mola_aux_loss_annealing: bool = field(default=False, metadata={"help": "Whether to anneal the aux loss."})
    mola_debug_mode: bool = field(default=False, metadata={"help": "Enable debug mode for MoLA."})
