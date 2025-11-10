import glob
import wandb
import random
import numpy as np
import torch

from transformers import TrainerCallback
import subprocess, json, os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transformers import set_seed

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LMEvalCallback(TrainerCallback):
    def __init__(self, model_path, eval_interval=500, eval_tasks=None, output_dir="./lm_eval_results", tb_logdir=None, batch_size=2, limit=500, cuda_devices="0", wandb_project="lm_eval_project", wandb_run_id=None):

        self.eval_interval = eval_interval
        self.eval_tasks = eval_tasks or ["belebele"]
        self.model_path = model_path
        self.wandb_run_id = wandb_run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=tb_logdir) if tb_logdir else None
        
        self.batch_size = batch_size
        self.limit = limit
        self.cuda_devices = cuda_devices
        self.wandb_project = wandb_project

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_interval != 0 or state.global_step == 0:
            return control

        try:
            step_dir = self.output_dir / f"step_{state.global_step}"
            step_dir.mkdir(parents=True, exist_ok=True)

            # start eval subprocess based on latest checkpoint
            model_path = self.get_latest_checkpoint(args.output_dir)
            
            run_env = os.environ.copy()
            run_env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
            run_env["WANDB_PROJECT"] = self.wandb_project
            run_env["WANDB_RUN_ID"] = self.wandb_run_id or ""
            run_env["WANDB_RESUME"] = "allow"
            subprocess.run([
                "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={self.model_path},peft={model_path},tokenizer={model_path}",
                "--tasks", ",".join(self.eval_tasks),
                "--batch_size", str(self.batch_size),
                "--limit", str(self.limit),
                "--output_path", str(step_dir),
                #"--wandb_args", f"project={self.wandb_project},group=eval,job_type=step_{state.global_step}",
                "--log_samples"
            ], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": self.cuda_devices})

            result_files = list(step_dir.glob("*/results_*.json"))
            if not result_files:
                raise FileNotFoundError(f"No results found in {step_dir}")

            with open(result_files[0]) as f:
                results = json.load(f)["results"]

            log_data = {}
            for task, metrics in results.items():
                print(f"[DEBUG] Task: {task}")
                for metric, value in metrics.items():
                    tag = f"{task}/{metric.replace(',', '_')}"
                    if isinstance(value, (int, float)):
                        log_data[tag] = value
                        if self.tb_writer:
                            self.tb_writer.add_scalar(tag, value, global_step=state.global_step)
                    print(f"[DEBUG] {tag} = {value}")
            wandb.log(log_data, step=state.global_step)
            
        except Exception as e:
            print(f"[LMEvalCallback] Evaluation step failed at global_step {state.global_step}: {e}")

        return control

    def get_latest_checkpoint(self, output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            print("[LMEvalCallback] No checkpoints found.")
            return output_dir
        print(f"[LMEvalCallback] Found checkpoints: {checkpoints} \nSelecting the latest one.")
        return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    def on_train_end(self, *args, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()