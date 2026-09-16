import torch
from pipeline_tools.cross_arch_verify import cast_inference_parameters


def test_inference_parameter_cast_preserves_rotary_buffer_precision():
    model = torch.nn.Linear(2, 2)
    frequencies = torch.tensor([0.123456789, 0.987654321], dtype=torch.float32)
    model.register_buffer("inv_freq", frequencies.clone())
    cast_inference_parameters(model)
    assert model.weight.dtype == torch.bfloat16
    assert model.inv_freq.dtype == torch.float32
    torch.testing.assert_close(model.inv_freq, frequencies, rtol=0, atol=0)


def test_mola_disable_adapter_bypasses_only_adapter_and_can_reenable():
    from peft.tuners.adamoe_1.layer import LinearMolaLayer
    torch.manual_seed(42)
    layer = LinearMolaLayer(torch.nn.Linear(4, 3), "default", lora_rank=2,
        lora_alpha=4, lora_dropout=0, num_experts=2, top_k=2).eval()
    with torch.no_grad():
        for expert in layer.lora_A["default"]:
            expert.weight.fill_(0.2)
        for expert in layer.lora_B["default"]:
            expert.weight.fill_(0.1)
        inputs = torch.ones(2, 4)
        base = layer.base_layer(inputs)
        enabled = layer(inputs)
        expected = base + layer.moe_layer["default"](inputs)
        torch.testing.assert_close(enabled, expected, rtol=0, atol=0)
        assert (enabled - base).abs().max() > 0
        layer.enable_adapters(False)
        torch.testing.assert_close(layer(inputs), base, rtol=0, atol=0)
        layer.enable_adapters(True)
        torch.testing.assert_close(layer(inputs), enabled, rtol=0, atol=0)
    for index, expert in enumerate(layer.moe_layer["default"].experts):
        assert expert.lora_A.weight is layer.lora_A["default"][index].weight
        assert expert.lora_B.weight is layer.lora_B["default"][index].weight
