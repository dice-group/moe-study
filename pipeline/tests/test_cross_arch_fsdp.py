import functools

import torch
from torch import nn
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, _recursive_wrap

from pipeline_tools.cross_arch_fsdp import shared_parameter_policy, local_gradient_status


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


def test_tied_endpoints_and_ancestors_stay_with_root_owner():
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.embedding = nn.Embedding(32, 8)
    model.backbone.block = nn.Linear(8, 8)
    model.lm_head = nn.Linear(8, 32, bias=False)
    model.lm_head.weight = model.backbone.embedding.weight
    weight = model.lm_head.weight
    original = functools.partial(size_based_auto_wrap_policy, min_num_params=1)
    policy, shared = shared_parameter_policy(model, original)
    assert shared == [["backbone.embedding.weight", "lm_head.weight"]]
    _recursive_wrap(model, policy, Wrapper, ignored_modules=set(), ignored_params=set(), only_wrap_children=True)
    assert not isinstance(model.backbone, Wrapper)
    assert not isinstance(model.backbone.embedding, Wrapper)
    assert not isinstance(model.lm_head, Wrapper)
    assert isinstance(model.backbone.block, Wrapper)
    assert model.lm_head.weight is model.backbone.embedding.weight is weight


def test_untied_model_keeps_original_size_policy():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    original = functools.partial(size_based_auto_wrap_policy, min_num_params=32)
    policy, shared = shared_parameter_policy(model, original)
    assert not shared
    for module in model.modules():
        for recurse in (True, False):
            n = sum(p.numel() for p in module.parameters())
            assert policy(module, recurse, n) == original(module, recurse, n)


def test_empty_local_gradient_shard_is_not_a_local_error():
    model = nn.Linear(2, 2, bias=False)
    assert local_gradient_status(model, "cpu").tolist() == [0, 0]
    model.weight.grad = torch.zeros_like(model.weight)
    assert local_gradient_status(model, "cpu").tolist() == [4, 0]
    model.weight.grad[0, 0] = float("nan")
    assert local_gradient_status(model, "cpu").tolist() == [4, 1]
