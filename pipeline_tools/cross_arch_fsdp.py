"""Keep shared parameters in the root FSDP owner, without changing model weights.

Gemma/Falcon tie the input embedding and output projection. Size-based wrapping
must not put either endpoint (or its ancestors) in a separate FSDP owner.
All other modules retain the original size-based policy.
"""
from collections import defaultdict
from contextlib import contextmanager
from types import MethodType


def install_checkpoint_safe_routing(model):
    """Campaign Trainer uses one forward/backward pair per microbatch.

    Recompute routing from the same IDs, rather than clearing IDs after forward
    or memoizing tensors created inside a checkpointed block. This leaves the
    routing formula and expert selection unchanged. IDs are replaced on the
    next microbatch; no autograd graph is retained in the cache.
    """
    from peft.tuners.moelora.model import MoELoraModel
    tuner = getattr(model, "base_model", None)
    if not isinstance(tuner, MoELoraModel):
        return False
    original_hooks = tuner._enable_peft_forward_hooks

    @contextmanager
    def hooks(self, *args, **kwargs):
        checkpointing = self.training and bool(getattr(self.model, "is_gradient_checkpointing", False))
        if not checkpointing:
            with original_hooks(*args, **kwargs):
                yield
            return
        self._cached_routing_ids = kwargs.get("language_ids")
        if self._cached_routing_ids is None:
            self._cached_routing_ids = kwargs.get("task_ids")
        self._cached_routing_weights = None
        with super(MoELoraModel, self)._enable_peft_forward_hooks(*args, **kwargs):
            yield

    tuner._enable_peft_forward_hooks = MethodType(hooks, tuner)
    return True


def local_gradient_status(model, device):
    """Return [owned gradient elements, nonfinite flag]; empty FSDP shards are valid."""
    import torch
    gradients = [p.grad for p in model.parameters()
                 if p.requires_grad and p.grad is not None and p.grad.numel()]
    elements = sum(gradient.numel() for gradient in gradients)
    bad = (~torch.stack([torch.isfinite(g).all() for g in gradients]).all()
           if gradients else torch.tensor(False, device=device))
    return torch.stack([torch.tensor(elements, device=device, dtype=torch.int64),
                        bad.to(device=device, dtype=torch.int64)])


def shared_parameter_policy(model, original_policy):
    owners = defaultdict(list)
    modules = dict(model.named_modules())
    for name, module in modules.items():
        for parameter_name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            owners[id(parameter)].append((name, parameter_name))
    shared = [locations for locations in owners.values() if len(locations) > 1]
    protected = set()
    for locations in shared:
        for name, _ in locations:
            pieces = name.split(".") if name else []
            protected.update(id(modules[".".join(pieces[:i])]) for i in range(len(pieces) + 1))

    def policy(module, recurse, nonwrapped_numel):
        if not recurse and id(module) in protected:
            return False
        return original_policy(module=module, recurse=recurse, nonwrapped_numel=nonwrapped_numel)

    return policy, [[f"{name}.{parameter}".lstrip(".") for name, parameter in group] for group in shared]


def install_shared_parameter_policy(model):
    from accelerate.state import AcceleratorState
    from types import MethodType
    plugin = getattr(AcceleratorState(), "fsdp_plugin", None)
    if plugin is None or getattr(plugin, "fsdp_version", 1) != 1:
        raise ValueError("Cross-architecture training requires the validated FSDP1 runtime")
    original_set = plugin.set_auto_wrap_policy

    def set_policy(self, actual_model):
        # Trainer may replace auto_wrap_policy after on_init_end. Reapply at
        # Accelerate's final prepare_model call, immediately before FSDP wraps.
        original_set(actual_model)
        if self.auto_wrap_policy is None:
            raise ValueError("Missing original FSDP auto-wrap policy")
        self.auto_wrap_policy, _ = shared_parameter_policy(actual_model, self.auto_wrap_policy)

    plugin.set_auto_wrap_policy = MethodType(set_policy, plugin)
    plugin.set_auto_wrap_policy(model)
    _, shared = shared_parameter_policy(model, plugin.auto_wrap_policy)
    if not shared:
        raise ValueError("Expected Gemma/Falcon tied embedding/output weights")
    return shared
