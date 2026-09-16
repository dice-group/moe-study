"""Normal LLaMA-Factory training with campaign validation and telemetry.

Pilots stop by SIGUSR1 at 120 steps without shortening the one-epoch LR
schedule. Production retains the original Slurm signal and compute budget.
"""
from __future__ import annotations

import json
import atexit
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from transformers import TrainerCallback

from pipeline_tools.cross_arch_study import write_json


def configure_allocator_limit():
    """Optional allocator-only bound; never changes precision, batch, or weights."""
    raw = os.environ.get("PROOF_PIPELINE_CUDA_MEMORY_FRACTION", "")
    if not raw:
        return
    fraction = float(raw)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Invalid CUDA allocator fraction")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)


class CampaignChecks(TrainerCallback):
    def __init__(self):
        self.timings = []
        self.gradient_checks = 0
        self.initial = {}
        self.stop_steps = int(os.environ.get("PROOF_PIPELINE_CROSS_ARCH_STOP_STEPS", "0"))

    def on_init_end(self, args, state, control, model=None, **kwargs):
        # Runtime RAM copies must never leak ephemeral paths into checkpoints.
        canonical = os.environ.get("PROOF_PIPELINE_CROSS_ARCH_CANONICAL_BASE")
        if canonical:
            model.config._name_or_path = canonical
            for config in model.peft_config.values():
                config.base_model_name_or_path = canonical
        from pipeline_tools.cross_arch_fsdp import install_shared_parameter_policy, install_checkpoint_safe_routing
        shared_parameters = install_shared_parameter_policy(model)
        checkpoint_safe_routing = install_checkpoint_safe_routing(model)
        target_names = set(next(iter(model.peft_config.values())).target_modules)
        modules = [name for name, module in model.named_modules()
                   if name.rsplit(".", 1)[-1] in target_names and hasattr(module, "lora_A")]
        layers = int(model.config.num_hidden_layers)
        if len(modules) != layers * len(target_names):
            raise ValueError(f"Adapter target coverage: expected {layers * len(target_names)}, got {len(modules)}")
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any(".base_layer." in name or name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight") for name, _ in trainable):
            raise ValueError("Unexpected trainable base-model parameters")
        if int(args.world_size) != 4 or int(args.per_device_train_batch_size) * int(args.gradient_accumulation_steps) * 4 != 48:
            raise ValueError("Campaign requires four GPUs and global batch 48")
        self.initial = {"targets": sorted(target_names), "adapted_modules": modules,
                        "fsdp_shared_parameters": shared_parameters,
                        "checkpoint_safe_routing": checkpoint_safe_routing,
                        "trainable_parameters": sum(p.numel() for _, p in trainable),
                        "total_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_names": [name for name, _ in trainable],
                        "microbatch": args.per_device_train_batch_size,
                        "gradient_accumulation": args.gradient_accumulation_steps,
                        "learning_rate": args.learning_rate, "optimizer": str(args.optim),
                        "scheduler": str(args.lr_scheduler_type), "warmup_steps": args.warmup_steps,
                        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
                        "trainer_gradient_checkpointing": args.gradient_checkpointing,
                        "world_size": args.world_size}
        self.initial["cuda_allocator_config"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        self.initial["cuda_memory_fraction"] = os.environ.get("PROOF_PIPELINE_CUDA_MEMORY_FRACTION", "")

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        self.last_step_end = self.started
        self.minimum_free_bytes = torch.cuda.mem_get_info()[0]
        self.training_peak_reserved = 0
        self.training_peak_allocated = 0
        self.initial["steps_per_epoch"] = state.max_steps
        torch.cuda.reset_peak_memory_stats()
        if state.is_world_process_zero:
            write_json(Path(args.output_dir) / "campaign_initial.json", self.initial)
            write_json(Path(args.output_dir) / "trainable_params.json", {
                "trainable_params": self.initial["trainable_parameters"], "all_params": self.initial["total_parameters"],
                "trainable_pct": 100 * self.initial["trainable_parameters"] / self.initial["total_parameters"],
                "source": "Pre-FSDP model parameter enumeration"})
            self.gpu_log = (Path(args.output_dir) / "gpu_utilization.csv").open("w")
            self.gpu_monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv", "-l", "10"], stdout=self.gpu_log, stderr=subprocess.DEVNULL)
            atexit.register(self.stop_gpu_monitor)

    def stop_gpu_monitor(self):
        monitor = getattr(self, "gpu_monitor", None)
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=5)
        log = getattr(self, "gpu_log", None)
        if log is not None and not log.closed:
            log.close()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.monotonic()

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if state.global_step != 0 and state.global_step % 50 != 0:
            return
        from pipeline_tools.cross_arch_fsdp import local_gradient_status
        # FSDP may leave a rank with no local adapter shard. Require gradients
        # globally and check every owned shard; never raise before the collective.
        status = local_gradient_status(model, torch.cuda.current_device())
        torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.SUM)
        if status[0].item() == 0:
            raise ValueError("No adapter gradients across any GPU")
        if status[1].item():
            raise ValueError("Non-finite adapter gradients")
        self.gradient_checks += 1

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.minimum_free_bytes = min(self.minimum_free_bytes, torch.cuda.mem_get_info()[0])
        self.training_peak_reserved = max(self.training_peak_reserved, torch.cuda.max_memory_reserved())
        self.training_peak_allocated = max(self.training_peak_allocated, torch.cuda.max_memory_allocated())
        now = time.monotonic()
        elapsed = now - self.last_step_end
        self.last_step_end = now
        if state.global_step > 20:
            self.timings.append(elapsed)
        if state.is_world_process_zero and state.global_step % 10 == 0:
            with (Path(args.output_dir) / "campaign_progress.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": state.global_step, "epoch": state.epoch,
                                         "trainer_seconds": time.monotonic() - self.started,
                                         "step_seconds": elapsed,
                                         "peak_reserved_bytes": torch.cuda.max_memory_reserved()}) + "\n")
        if self.stop_steps and state.global_step >= self.stop_steps:
            # The existing JIT callback follows this callback and sees this signal.
            os.kill(os.getpid(), signal.SIGUSR1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm"):
            if key in (logs or {}) and not math.isfinite(float(logs[key])):
                raise ValueError(f"Non-finite training {key}")

    def on_train_end(self, args, state, control, **kwargs):
        # Preserve the overall gate; expose phase peaks to diagnose checkpoint overhead.
        self.minimum_free_bytes = min(self.minimum_free_bytes, torch.cuda.mem_get_info()[0])
        phases = torch.tensor([self.training_peak_reserved, self.training_peak_allocated,
                               torch.cuda.max_memory_allocated()], dtype=torch.int64,
                              device=torch.cuda.current_device())
        torch.distributed.all_reduce(phases, op=torch.distributed.ReduceOp.MAX)
        memory = torch.tensor([torch.cuda.max_memory_reserved(), torch.cuda.get_device_properties(0).total_memory, self.minimum_free_bytes],
                              dtype=torch.int64, device=torch.cuda.current_device())
        torch.distributed.all_reduce(memory[:1], op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(memory[2:], op=torch.distributed.ReduceOp.MIN)
        if state.is_world_process_zero:
            self.stop_gpu_monitor()
            write_json(Path(args.output_dir) / "campaign_training.json", {
                **self.initial, "global_step": state.global_step, "epoch": state.epoch,
                "gradient_checks": self.gradient_checks, "step_seconds_after_warmup": self.timings,
                "peak_reserved_bytes": int(memory[0]), "gpu_memory_bytes": int(memory[1]),
                "minimum_free_bytes": int(memory[2]),
                "training_peak_reserved_bytes": int(phases[0]),
                "training_peak_allocated_bytes": int(phases[1]),
                "overall_peak_allocated_bytes": int(phases[2]),
                "signal_checkpoint": bool(getattr(args, "jit_checkpoint_signal_stop", False)),
                "pilot": bool(self.stop_steps), "trainer_seconds": time.monotonic() - self.started})


def main():
    configure_allocator_limit()
    # Keep filter-generated Arrow caches outside the verified tokenized input.
    # Otherwise each retry sees extra cache-*.arrow files and restages all data.
    # Filtering, row order, examples and the one-epoch schedule are unchanged.
    from datasets import disable_caching
    disable_caching()
    from pipeline_tools.cross_arch_verify import verify_runtime
    verify_runtime()
    if "--model_name_or_path" in sys.argv:
        from pipeline_tools.cross_arch_model_cache import local_model_path
        from pipeline_tools.cross_arch_study import contract, model_root
        position = sys.argv.index("--model_name_or_path") + 1
        for model_name in contract()["models"]:
            canonical = model_root(model_name) / "model"
            if Path(sys.argv[position]) == canonical:
                local = local_model_path(model_name)
                if local != canonical:
                    os.environ["PROOF_PIPELINE_CROSS_ARCH_CANONICAL_BASE"] = str(canonical)
                    sys.argv[position] = str(local)
                    print(f"[MODEL_CACHE] loading verified copy {local}; checkpoint base stays {canonical}", flush=True)
                break
    if os.environ.get("PROOF_PIPELINE_CROSS_ARCH_PRODUCTION") == "1":
        from pipeline_tools.cross_arch_launch import validate_gate
        if os.environ.get("PROOF_PIPELINE_CROSS_ARCH_STOP_STEPS") or os.environ.get("HALA_MAX_STEPS_OVERRIDE"):
            raise ValueError("Production must not inherit pilot/max-step overrides")
        validate_gate(os.environ["PROOF_PIPELINE_CROSS_ARCH_MODEL"], os.environ["PROOF_PIPELINE_CROSS_ARCH_RUN"])
    from llamafactory.train.tuner import run_exp
    run_exp(callbacks=[CampaignChecks()])


if __name__ == "__main__":
    main()
