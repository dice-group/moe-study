import copy
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from peft import MoELoraConfig, TaskType, get_peft_model
from pipeline_tools.cross_arch_fsdp import install_checkpoint_safe_routing


def test_checkpointed_routing_matches_losses_and_all_adapter_gradients():
    torch.manual_seed(42)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, use_cache=False))
    reference = get_peft_model(base, MoELoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        lora_dropout=0, expert_num=2, task_num=3, task_embedding_dim=4, gate_top_k=1,
        target_modules=["q_proj", "v_proj"]))
    for name, parameter in reference.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.02)
    checkpointed = copy.deepcopy(reference)
    # Rebind weak parent references after deepcopy, as normal construction does.
    from peft.tuners.moelora.layer import MoELoraLayer
    for module in checkpointed.modules():
        if isinstance(module, MoELoraLayer):
            module.set_moelora_parent(checkpointed.base_model)
    assert install_checkpoint_safe_routing(checkpointed)
    reference.train()
    checkpointed.train()
    checkpointed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for language in (1, 2):
        inputs = torch.tensor([[1, 2, 3, 4]])
        batch = {"input_ids": inputs, "labels": inputs, "language_ids": torch.tensor([language])}
        reference.zero_grad(set_to_none=True)
        checkpointed.zero_grad(set_to_none=True)
        expected, observed = reference(**batch).loss, checkpointed(**batch).loss
        expected.backward()
        observed.backward()
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        actual = dict(checkpointed.named_parameters())
        for name, parameter in reference.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                assert actual[name].grad is not None, name
                torch.testing.assert_close(actual[name].grad, parameter.grad, rtol=1e-4, atol=1e-6, msg=name)
