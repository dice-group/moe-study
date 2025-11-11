import glob
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import wandb
from transformers import TrainerCallback, set_seed

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LMEvalCallback(TrainerCallback):
    def __init__(self, model_path, eval_interval=500, eval_tasks=None, batch_size=2, limit=500, cuda_devices="0", wandb_project="lm_eval_project", wandb_run_id=None,
    ):
        self.eval_interval = max(eval_interval, 1)
        self.eval_tasks = eval_tasks or ["belebele"]
        self.model_path = model_path
        self.batch_size = batch_size
        self.limit = limit
        self.cuda_devices = cuda_devices
        self.wandb_project = wandb_project
        self.wandb_run_id = wandb_run_id
        self._last_eval_step = 0

    def _should_run(self, state):
        step = state.global_step
        if step == 0:
            return False
        if step == self._last_eval_step:
            return False
        return step % self.eval_interval == 0

    def on_save(self, args, state, control, **kwargs):
        if not self._should_run(state):
            return control
        self._last_eval_step = state.global_step
        self._run_eval(args, state)
        return control

    def _run_eval(self, args, state):
        try:
            model_path = self.get_latest_checkpoint(args.output_dir)

            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": self.cuda_devices,
                "WANDB_PROJECT": self.wandb_project,
                "WANDB_RUN_ID": self.wandb_run_id or "",
                "WANDB_RESUME": "allow",
            }

            with tempfile.TemporaryDirectory(prefix=f"lm_eval_step_{state.global_step}_") as tmpdir:
                tmp_path = Path(tmpdir)
                subprocess.run(
                    [
                        "lm_eval",
                        "--model", "hf",
                        "--model_args", f"pretrained={self.model_path},peft={model_path},tokenizer={model_path}",
                        "--tasks", ",".join(self.eval_tasks),
                        "--batch_size", str(self.batch_size),
                        "--limit", str(self.limit),
                        "--output_path", str(tmp_path),
                        "--log_samples",
                    ],
                    check=True,
                    env=env,
                )

                result_files = list(tmp_path.glob("*/results_*.json"))
                if not result_files:
                    raise FileNotFoundError(f"No results found in temporary directory for step {state.global_step}")

                with open(result_files[0]) as f:
                    results = json.load(f)["results"]

            log_data = {}
            for task, metrics in results.items():
                for metric, value in metrics.items():
                    tag = f"{task}/{metric.replace(',', '_')}"
                    if isinstance(value, (int, float)):
                        log_data[tag] = value
            if log_data:
                wandb.log(log_data, step=state.global_step)

        except Exception as e:
            print(f"[LMEvalCallback] Evaluation step failed at global_step {state.global_step}: {e}")

    def get_latest_checkpoint(self, output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            print("[LMEvalCallback] No checkpoints found.")
            return output_dir
        print(f"[LMEvalCallback] Found checkpoints: {checkpoints} \nSelecting the latest one.")
        return max(checkpoints, key=lambda x: int(x.split("-")[-1]))
