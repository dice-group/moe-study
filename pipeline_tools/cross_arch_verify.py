"""Real-weight attention-backend and adapter round-trip validation on an A100."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

from pipeline_tools.cross_arch_study import ROOT, contract, model_root, read_json, write_json, utc_now, PYTHON, checkpoint_weights


def cast_inference_parameters(model):
    """Match Factory inference without reducing rotary-buffer precision."""
    import torch
    for parameter in model.parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.eval()


def verify_runtime():
    import peft
    import llamafactory
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise ValueError("Wrong Python environment")
    sources = {"peft": peft.__file__, "llamafactory": llamafactory.__file__}
    for name, path in sources.items():
        if not Path(path).resolve().is_relative_to((ROOT / "LLaMA-Factory/src" / name).resolve()):
            raise ValueError(f"Wrong {name} checkout imported: {path}")
    return sources


def validate_backend(model_name):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sources = verify_runtime()
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("Backend validation requires an allocated A100")
    out = model_root(model_name)
    assets = read_json(out / "assets.json")
    tokenizer = AutoTokenizer.from_pretrained(out / "model", trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(out / "model", torch_dtype=torch.bfloat16,
                                                device_map={"": 0}, attn_implementation="eager", trust_remote_code=False).eval()
    requested = "flash_attention_2" if model_name == "gemma2_9b" else "sdpa"
    text = "A multilingual study. Das ist ein Beispielsatz. Это пример. هذا مثال. 这是一个例子。 " * 1000
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    lengths = [128, 1024, 4097] if model_name == "gemma2_9b" else [128, 1024, 2048]
    reference = {}
    with torch.no_grad():
        for length in lengths:
            ids = encoded[:, :length].to("cuda")
            extra = {"logits_to_keep": 1} if model_name == "gemma2_9b" else {}
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **extra).logits[:, -1].float()
            if not torch.isfinite(output).all():
                raise ValueError("Non-finite eager reference")
            reference[length] = output.cpu()
    selected, error, differences = "disabled", None, {}
    try:
        model.set_attn_implementation(requested)
        with torch.no_grad():
            for length in lengths:
                ids = encoded[:, :length].to("cuda")
                extra = {"logits_to_keep": 1} if model_name == "gemma2_9b" else {}
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, **extra).logits[:, -1].float().cpu()
                torch.testing.assert_close(output, reference[length], rtol=0.02, atol=0.08)
                differences[length] = float((output-reference[length]).abs().max())
        selected = "fa2" if requested == "flash_attention_2" else "sdpa"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Eager reference has already passed at every required length.
    result = {"pass": True, "model": model_name, "revision": assets["revision"], "selected_backend": selected,
              "accelerated_backend": requested, "accelerated_backend_error": error,
              "max_logit_differences": differences, "lengths": lengths,
              "torch": torch.__version__, "transformers": transformers.__version__, "python": sys.executable,
              "gpu": torch.cuda.get_device_name(0), "sources": sources, "checked_utc": utc_now()}
    if selected == "fa2":
        import flash_attn_2_cuda
        from pipeline_tools.cross_arch_study import sha256
        result["flash_attention_binary"] = str(Path(flash_attn_2_cuda.__file__).resolve())
        result["flash_attention_sha256"] = sha256(result["flash_attention_binary"])
    write_json(out / "backend_ready.json", result)
    print(json.dumps(result, indent=2), flush=True)


def verify_adapter(model_name, checkpoint):
    verify_runtime()
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, get_peft_model_state_dict
    from datasets import load_from_disk
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    out = model_root(model_name)
    from pipeline_tools.cross_arch_model_cache import local_model_path
    base_path = local_model_path(model_name)
    checkpoint = Path(checkpoint)
    adapter_config = read_json(checkpoint / "adapter_config.json")
    backend = read_json(out / "backend_ready.json")["selected_backend"]
    attn = {"disabled": "eager", "fa2": "flash_attention_2", "sdpa": "sdpa"}[backend]
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                                               device_map={"": 0}, attn_implementation=attn, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False).eval()
    weights = checkpoint_weights(checkpoint)
    saved = load_file(str(weights)) if weights.suffix == ".safetensors" else torch.load(weights, map_location="cpu", weights_only=True)
    saved_tensor_count = len(saved)
    loaded = get_peft_model_state_dict(model)
    for key, tensor in saved.items():
        if not torch.isfinite(tensor).all() or key not in loaded:
            raise ValueError(f"Missing/nonfinite checkpoint tensor: {key}")
        torch.testing.assert_close(loaded[key].float().cpu(), tensor.float(), rtol=0, atol=0)
    row = load_from_disk(str(out / "tokenized_ranks/rank_00000"))[0]
    ids = torch.tensor([row["input_ids"][:32]], device="cuda")
    mask = torch.tensor([row["attention_mask"][:32]], device="cuda")
    language_ids = torch.tensor([row["language_ids"]], device="cuda")
    with torch.no_grad():
        reference = model(input_ids=ids, attention_mask=mask, language_ids=language_ids, labels=ids)
        logits, loss = reference.logits.float().cpu(), float(reference.loss)
        with model.disable_adapter():
            plain = model(input_ids=ids, attention_mask=mask, language_ids=language_ids).logits.float().cpu()
    delta = float((logits-plain).abs().max())
    native_delta, effect_precision = delta, "native"
    if delta <= 1e-6 and torch.isfinite(logits).all():
        # A three-step adapter can be smaller than BF16 rounding. Check its
        # functional effect in FP32; benchmark/round-trip inference stays BF16.
        # Stage the conversion through CPU to avoid two model copies on GPU.
        model = model.cpu().float()
        torch.cuda.empty_cache()
        model = model.to("cuda").eval()
        with torch.no_grad():
            precise = model(input_ids=ids, attention_mask=mask, language_ids=language_ids).logits.float().cpu()
            with model.disable_adapter():
                plain = model(input_ids=ids, attention_mask=mask, language_ids=language_ids).logits.float().cpu()
        delta = float((precise-plain).abs().max())
        effect_precision = "float32"
        if not torch.isfinite(precise).all():
            raise ValueError("Nonfinite FP32 adapter-effect diagnostic")
    if not torch.isfinite(logits).all() or not delta > 1e-6:
        raise ValueError("Reloaded adapter has no observable trained effect")
    # Match the Factory inference path exactly: it merges standard LoRA, then
    # casts remaining FP32 parameters to BF16. Comparing against FP32 adapters
    # (PEFT's default reload dtype) is not an apples-to-apples round-trip check.
    kind = str(adapter_config["peft_type"]).lower()
    kind = {"adamoe_1": "mola"}.get(kind, kind)
    if kind == "lora":
        model = model.merge_and_unload()
    # Do not cast buffers: rotary frequencies must retain their FP32 precision.
    cast_inference_parameters(model)
    with torch.no_grad():
        reference = model(input_ids=ids, attention_mask=mask, language_ids=language_ids, labels=ids)
        logits, loss = reference.logits.float().cpu(), float(reference.loss)
    del reference, model, base, loaded, saved
    gc.collect()
    torch.cuda.empty_cache()
    ma, da, fa, _ = get_infer_args({"model_name_or_path": str(base_path), "adapter_name_or_path": str(checkpoint),
                                  "finetuning_type": kind, "template": "empty", "infer_dtype": "bfloat16",
                                  "flash_attn": backend, "trust_remote_code": False})
    tokenizer = load_tokenizer(ma)["tokenizer"]
    get_template_and_fix_tokenizer(tokenizer, da)
    factory = load_model(tokenizer, ma, fa, is_trainable=False).eval().to("cuda")
    with torch.no_grad():
        output = factory(input_ids=ids, attention_mask=mask, language_ids=language_ids, labels=ids)
    # Standard LoRA may be merged by the factory; allow BF16 rounding, not lost adapters.
    torch.testing.assert_close(output.logits.float().cpu(), logits, rtol=0.02, atol=0.08)
    result = {"pass": True, "checkpoint": str(checkpoint), "saved_tensor_count": saved_tensor_count,
              "max_trained_delta": delta, "native_precision_delta": native_delta,
              "effect_check_precision": effect_precision, "direct_loss": loss, "factory_loss": float(output.loss),
              "factory_max_logit_difference": float((output.logits.float().cpu()-logits).abs().max())}
    write_json(checkpoint.parent / "reload_check.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["backend", "adapter"])
    parser.add_argument("--model", required=True, choices=list(contract()["models"]))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.action == "backend": validate_backend(args.model)
    else:
        if args.checkpoint is None: parser.error("--checkpoint required")
        verify_adapter(args.model, args.checkpoint)


if __name__ == "__main__":
    main()
