# Slurm Time-Limit Checkpointing (Findings)

## What We Observed
- The main problem was not just "buffer too small".
- In the original `submit-chain` wiring, Slurm submitted `python -m pipeline.run --executor-type local`, which then launched `llamafactory.cli train`.
- That extra `local` executor hop made the warning-signal path brittle: the Slurm signal hit the wrapper layer first, not the actual trainer process.
- The Slurm batch wrapper also used `set -e`, so when `wait "$CHILD_PID"` was interrupted by `SIGUSR1`, the shell exited immediately before the trainer could finish checkpointing.

## Key Insight
Two things were required:
- The pre-signal must reach the actual trainer process directly.
- The Slurm wrapper must survive an interrupted `wait` long enough for the trainer to save and exit cleanly.

After that, normal walltime slack matters again: the warning still needs to come early enough that checkpoint writing can finish.

## Resolution
- `submit-chain` now submits `train` jobs directly as `llamafactory.cli train` instead of nesting them under `pipeline.run --executor-type local`.
- The Slurm wrapper now `exec`s the real child process and uses a signal-safe wait loop instead of exiting on interrupted `wait`.
- With this change, the smoke test works with a real Slurm warning signal:
  - job time: `00:06:00`
  - warning: `--signal=B:USR1@300`
  - result: trainer receives `SIGTERM`, saves `checkpoint-*` and `checkpoint-*_adapter`, and the job exits `COMPLETED`.

## Practical Recommendation
- Keep the direct Slurm-to-trainer path for `train` jobs in `submit-chain`.
- Use a pre-signal with meaningful slack before the hard limit.
  - For real runs, choose the buffer based on measured checkpoint time, e.g. `@300`, `@900`, or `@1800`.
- Validate on each approach by checking that the trainer log shows signal receipt, checkpoint request, and checkpoint completion.

## Debugging Checklist
- Confirm the trainer actually receives the signal by grepping the train log for:
  - `Signal received at unix=`
  - `Received SIGTERM:`
  - `Signal: requesting checkpoint save`
  - `Signal: checkpoint saved`
- If none of these lines appear, the signal is still being lost in the wrapper/executor path.
- If the request appears but the save does not finish, the remaining issue is usually **insufficient buffer**.
