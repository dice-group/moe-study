###  TODOs

- Script that takes a data_sample, model and extra config file to schedule a training & evaluation job on the cluster
- We should try to limit to models that fit on single A100
    - Use quantized versions that fit: https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF
    - Take runtime in consideration
- Measuring FLOPs would be nice, it provides extra depth to analysis
- Connect wandb to track training progress
- Should have uniform output_paths, analysis, evaluation outputs