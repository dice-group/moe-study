from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor checkpoints and trigger lm-eval jobs.")
    parser.add_argument("--watch-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--eval-script", required=True)
    parser.add_argument("--eval-log-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-name-prefix", default="")
    parser.add_argument("--tasks", default="belebele")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--poll-interval", type=float, default=120.0)
    parser.add_argument("--idle-timeout", type=float, default=60.0, help="Minutes without new checkpoints before exiting.")
    parser.add_argument("--max-duration", type=float, default=1440.0, help="Maximum minutes to watch before exiting.")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--eval-partition", default="")
    parser.add_argument("--eval-time", default="")
    parser.add_argument("--eval-gpus", type=int, default=1)
    parser.add_argument("--eval-gpu-type", default="")
    parser.add_argument("--eval-extra-sbatch", default="")
    return parser.parse_args()


def submit_eval_job(
    ckpt_path: Path,
    args: argparse.Namespace,
) -> None:
    ckpt_name = ckpt_path.name
    timestamp = int(time.time())
    env = {
        "CHECKPOINT_PATH": str(ckpt_path),
        "TOKENIZER_PATH": args.tokenizer_path,
        "LM_EVAL_TASKS": args.tasks,
        "LM_EVAL_BATCH_SIZE": args.batch_size,
        "LM_EVAL_RESULTS_DIR": args.results_dir or str(ckpt_path / "lm_eval_results"),
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_GROUP": args.wandb_group or args.label,
        "WANDB_NAME": f"{args.wandb_name_prefix or args.label}-{ckpt_name}-lm-eval",
    }
    env_assignments = [f"{k}={v}" for k, v in env.items() if v is not None]
    log_dir = Path(args.eval_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"lm_eval_{ckpt_name}_{timestamp}.log"
    cmd = ["env", *env_assignments, "sbatch", f"--output={log_path}"]
    if args.eval_partition:
        cmd.append(f"--partition={args.eval_partition}")
    if args.eval_time:
        cmd.append(f"--time={args.eval_time}")
    if args.eval_gpus is not None:
        if args.eval_gpu_type:
            cmd.append(f"--gres=gpu:{args.eval_gpu_type}:{args.eval_gpus}")
        else:
            cmd.append(f"--gres=gpu:{args.eval_gpus}")
    if args.eval_extra_sbatch:
        cmd.extend(shlex.split(args.eval_extra_sbatch))
    cmd.append(args.eval_script)
    subprocess.run(cmd, check=True)
    print(f"[INFO] Launched lm-eval for {ckpt_path}")


def main() -> None:
    args = parse_args()
    watch_dir = Path(args.watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    start_ts = time.time()
    last_seen_ts = start_ts
    idle_timeout = args.idle_timeout * 60.0
    max_duration = args.max_duration * 60.0
    while True:
        ckpts = sorted(
            p for p in watch_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint")
        )
        for ckpt in ckpts:
            if ckpt not in seen:
                seen.add(ckpt)
                submit_eval_job(ckpt, args)
                last_seen_ts = time.time()
        now = time.time()
        if now - start_ts > max_duration:
            print("[INFO] Max watch duration reached; exiting listener.")
            break
        if now - last_seen_ts > idle_timeout:
            print("[INFO] Idle timeout reached with no new checkpoints; exiting listener.")
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
