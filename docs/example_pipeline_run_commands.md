## Example run commands for 
Example run commands for the pipeline for quick testing / verification

1. On GPU VM
```bash
HALA_MAX_STEPS_OVERRIDE=100 \
  /local/joeldag/miniconda3/envs/merlin/bin/python -m pipeline.run \
    --config configs/pipeline \
    --profile moe_study_profile_lola_gpuvm_mixlora_smoke \
    --run-id moelora \
    --executor-type local \
    --cuda-visible-devices 0 \
    --start-at sample \
    --stop-after report
```

to run detached + tail 
```bash
nohup bash -lc '
  cd /local/joeldag/repos/moe-study && \
  HALA_MAX_STEPS_OVERRIDE=100 \
  /local/joeldag/miniconda3/envs/merlin/bin/python -m pipeline.run \
    --config configs/pipeline \
    --profile moe_study_profile_lola_gpuvm_mixlora_smoke \
    --run-id moelora \
    --executor-type local \
    --cuda-visible-devices 0 \
    --start-at sample \
    --stop-after report
' > /local/joeldag/repos/moe-study/outputs/pipeline_runs/moe_study_profile_lola_gpuvm_mixlora_smoke/nohup.out 2>&1 &

tail -f /local/joeldag/repos/moe-study/outputs/pipeline_runs/moe_study_profile_lola_gpuvm_mixlora_smoke/nohup.out
```


## for otus cluster
for quick verification, run two approaches from tokenization -> until report to see a full gothrough of the pipeline

```
HALA_MAX_STEPS_OVERRIDE=10 LM_EVAL_LIMIT=10 \
python -m pipeline.run --config configs/pipeline \
  --profile moe_study_profile_otus_cluster_96_langs \
  --run-id hmora \
  --run-id lora_r64 \
  --start-at tokenize --stop-after report \
  --submit-chain
```