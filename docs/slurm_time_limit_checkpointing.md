# Slurm Time-Limit Checkpointing (Findings)

## What We Observed
- Slurm time-limit termination can kill the job before the trainer finishes saving a checkpoint.
- When the signal *does* reach the trainer, we see logs like:
  - `Signal received at unix=...: SIGTERM`
  - `Received SIGTERM: will save a checkpoint at next step end and stop training.`
- In failing runs, the job hits the hard time limit immediately after the signal, so there is no time left to finish `checkpoint-*` (and `checkpoint-*_adapter`) writing.

## Key Insight
We need **guaranteed walltime slack after the pre-signal**.
If the signal is delivered too close to the hard limit, saving will not complete.

## Practical Recommendation (Cluster)
- Use `SIGTERM` as the warning signal (not `SIGUSR1`).
- Make the warning happen **well before** the hard limit.
  - For real runs: prefer something like `--signal=B:TERM@1800` (30 min buffer) depending on checkpoint size.
- Ensure the stage time limit (`profile.training.executor.time`) is long enough that the buffer is meaningful.

## Debugging Checklist
- Confirm the trainer actually receives the signal by grepping the train log for:
  - `Signal received at unix=`
  - `Received SIGTERM:`
  - `Signal: requesting checkpoint save`
  - `Signal: checkpoint saved`
- If the first two lines appear but no checkpoint is written, it is almost always **insufficient buffer**.
