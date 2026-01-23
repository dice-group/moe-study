# SLURM test notes

## I/O copy test (2026-01-23)
- Script: `io_copy_test.sh`
- Log: `scripts/slurm/io_copy_test_result.log`
- Baseline (shared FS): 361s
- Copy time: 334s
- Local copy run: 358s

Conclusion: local copy did not improve runtime (within noise). Copying adds overhead; not needed unless reusing the local copy for many runs on the same node.
