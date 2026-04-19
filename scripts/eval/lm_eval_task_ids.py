#!/usr/bin/env python3
"""Run lm-eval with an optional fixed task_ids tensor injected into model calls."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from lm_eval.tasks import TaskManager


LOGGER = logging.getLogger("lm_eval_task_ids")
_MODEL_CACHE: dict[tuple[str, str, str, str], object] = {}


def _parse_torch_dtype(value: Optional[str]):
    if value is None:
        return None
    val = str(value).strip().lower()
    if val in {"", "none", "null"}:
        return None
    if val == "auto":
        return "auto"
    if val in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if val in {"fp16", "float16", "half"}:
        return torch.float16
    if val in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def _normalize_device_map(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    val = str(value).strip()
    if not val or val.lower() in {"none", "null"}:
        return None
    return val


def _resolve_adapter_dir(path: Path) -> Path:
    if (path / "adapter_config.json").exists():
        return path
    alt = Path(f"{path}_adapter")
    if (alt / "adapter_config.json").exists():
        return alt
    return path


def _load_base_model_name(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return ""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return str(cfg.get("base_model_name_or_path", "")).strip()


def _parse_tasks(raw: str) -> list[str]:
    if os.path.isfile(raw):
        return [line.strip() for line in Path(raw).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [task.strip() for task in raw.split(",") if task.strip()]


def _build_hflm_with_task_class(HFLM):
    class HFLMWithTask(HFLM):
        def __init__(self, task_id: Optional[int] = None, **kwargs):
            self._task_id = task_id
            super().__init__(**kwargs)

        def _inject_task_ids(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
            if self._task_id is None:
                return None
            return torch.full(
                (input_ids.size(0),),
                int(self._task_id),
                device=input_ids.device,
                dtype=torch.long,
            )

        def _model_call(self, inps, attn_mask=None, labels=None):
            import transformers

            task_ids = self._inject_task_ids(inps)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.mixed_precision_dtype,
                    enabled=self.mixed_precision_dtype is not None,
                ),
            ):
                if attn_mask is not None or labels is not None:
                    assert attn_mask is not None and labels is not None
                    assert transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS
                    kwargs = {"task_ids": task_ids} if task_ids is not None else {}
                    return self.model(
                        input_ids=inps,
                        attention_mask=attn_mask,
                        labels=labels,
                        **kwargs,
                    ).logits

                assert self.AUTO_MODEL_CLASS in (
                    transformers.AutoModelForCausalLM,
                    transformers.AutoModelForVision2Seq,
                )
                if task_ids is not None:
                    return self.model(input_ids=inps, task_ids=task_ids).logits
                return self.model(inps).logits

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            task_ids = self._inject_task_ids(context)
            if task_ids is not None:
                generation_kwargs["task_ids"] = task_ids
            return super()._model_generate(context, max_length, stop, **generation_kwargs)

    return HFLMWithTask


def _run_eval(
    *,
    pretrained: str,
    peft: str,
    tokenizer: str,
    tasks: list[str],
    batch_size: str,
    output_path: Path,
    device: str,
    task_id: Optional[int],
    limit: Optional[float],
    include_path: Optional[str],
    torch_dtype,
    device_map: Optional[str],
    seed: int,
    numpy_seed: int,
    torch_seed: int,
    fewshot_seed: int,
) -> None:
    repo_root = None
    for parent in Path(__file__).resolve().parents:
        if (parent / "LLaMA-Factory" / "src" / "peft").exists():
            repo_root = parent
            break
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    peft_src = repo_root / "LLaMA-Factory" / "src"
    if str(peft_src) not in sys.path:
        sys.path.insert(0, str(peft_src))
    for name in list(sys.modules.keys()):
        if name == "peft" or name.startswith("peft."):
            del sys.modules[name]

    import importlib.util

    peft_pkg = peft_src / "peft"
    peft_init = peft_pkg / "__init__.py"
    spec = importlib.util.spec_from_file_location("peft", peft_init, submodule_search_locations=[str(peft_pkg)])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local peft from {peft_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["peft"] = module
    spec.loader.exec_module(module)

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    task_manager = TaskManager(include_path=include_path)
    model_key = (
        pretrained,
        peft,
        str(device_map) if device_map is not None else "none",
        str(torch_dtype) if torch_dtype is not None else "none",
    )
    if model_key not in _MODEL_CACHE:
        load_kwargs = {"low_cpu_mem_usage": True}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        base_model = AutoModelForCausalLM.from_pretrained(pretrained, **load_kwargs)
        peft_model = PeftModel.from_pretrained(base_model, peft, is_trainable=False)
        if device.startswith("cuda") and torch.cuda.is_available() and device_map is None:
            peft_model = peft_model.to(device)
        peft_model.eval()
        _MODEL_CACHE[model_key] = peft_model

    HFLMWithTask = _build_hflm_with_task_class(HFLM)
    model = HFLMWithTask(
        task_id=task_id,
        pretrained=_MODEL_CACHE[model_key],
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
        task_manager=task_manager,
        random_seed=seed,
        numpy_random_seed=numpy_seed,
        torch_random_seed=torch_seed,
        fewshot_random_seed=fewshot_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--include-path", default=None)
    parser.add_argument("--torch-dtype", default=os.environ.get("LM_EVAL_TORCH_DTYPE"))
    parser.add_argument("--device-map", default=os.environ.get("LM_EVAL_DEVICE_MAP"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LM_EVAL_RANDOM_SEED", 42)))
    parser.add_argument("--numpy-seed", type=int, default=int(os.environ.get("LM_EVAL_NUMPY_SEED", 42)))
    parser.add_argument("--torch-seed", type=int, default=int(os.environ.get("LM_EVAL_TORCH_SEED", 42)))
    parser.add_argument("--fewshot-seed", type=int, default=int(os.environ.get("LM_EVAL_FEWSHOT_SEED", 42)))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s|%(asctime)s|%(message)s")
    checkpoint = _resolve_adapter_dir(Path(args.checkpoint))
    base_model = _load_base_model_name(checkpoint) or args.tokenizer
    tasks = _parse_tasks(args.tasks)
    torch_dtype = _parse_torch_dtype(args.torch_dtype)
    if torch_dtype is None and args.device.startswith("cuda"):
        torch_dtype = "auto"
    _run_eval(
        pretrained=base_model,
        peft=str(checkpoint),
        tokenizer=base_model,
        tasks=tasks,
        batch_size=args.batch_size,
        output_path=Path(args.output_path),
        device=args.device,
        task_id=args.task_id,
        limit=args.limit,
        include_path=args.include_path,
        torch_dtype=torch_dtype,
        device_map=_normalize_device_map(args.device_map),
        seed=args.seed,
        numpy_seed=args.numpy_seed,
        torch_seed=args.torch_seed,
        fewshot_seed=args.fewshot_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
