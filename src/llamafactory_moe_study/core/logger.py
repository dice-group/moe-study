import logging
import os

def get_logger(name: str = "adamole"):
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

log = get_logger()


def log_adapter_architecture(model):
    """
    Logs a minimal summary of AdaMoLE or MoLA adapters.
    """
    from llamafactory_moe_study.approaches.adamole.adamole.layer import LinearAdaMoleLayer
    from llamafactory_moe_study.approaches.mola.layer import LinearMolaLayer

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info(f"Adapter model: {trainable_params:,d} trainable / {total_params:,d} total parameters")

    for name, mod in model.named_modules():
        if isinstance(mod, LinearAdaMoleLayer):
            active = mod._active_adapter

            # Handle both single and multiple active adapters
            if isinstance(active, (list, tuple)):
                for idx in active:
                    log.info(
                        f" - {name}[{idx}] (AdaMoLE): {len(mod.moe_layer[idx].experts)} experts, "
                        f"rank={mod.lora_rank[idx]}, "
                        f"τ_max={mod.moe_layer[idx].max_threshold}"
                    )
            else:
                log.info(
                    f" - {name} (AdaMoLE): {len(mod.moe_layer[active].experts)} experts, "
                    f"rank={mod.lora_rank[active]}, "
                    f"τ_max={mod.moe_layer[active].max_threshold}"
                )
        elif isinstance(mod, LinearMolaLayer):
            active = mod._active_adapter
            log.info(
                f" - {name} (MoLA): {len(mod.moe_layer[active].experts)} experts, "
                f"top_k={mod.moe_layer[active].top_k}, "
                f"rank={mod.lora_rank[active]}"
            )
