# MoE Study 


### Overview of approaches

| Integrated | Model     | Framework Base             | Training Type               | Libraries Used                          | Model Finetuned / Trained                                           |
|------------|-----------|----------------------------|-----------------------------|-----------------------------------------|---------------------------------------------------------------------|
| [ ]        | Ada_MoE_1 | MoLA (Mixture of Adapter)  | Adapter (LoRA) fine-tuning  | HuggingFace Transformers + PEFT         | Llama2-7B                                                           |
| [x]        | Ada_MoE_2 | LLaMA Factory              | Adapter (QLoRA) fine-tuning | LLaMA Factory                           | Mixtral-8x7B                                                        |
| [ ]        | AdaMoLE   | HuggingFace setup          | Adapter (LoRA) fine-tuning  | HuggingFace Transformers + PEFT         | Llama2-7B                                                           |
| [x]        | MoE-LPR   | LLaMA Factory              | Full fine-tuning            | LLaMA Factory                           | Qwen-1.5-1.8B                                                       |
| [ ]        | XMoE      | Megatron-DeepSpeed         | Pretraining                 | Megatron-LM + DeepSpeed                 | Custom: 12-layer decoder-only Transformer (GPT-2 tokenizer)         |
| [ ]        | XMoE_2    | Fairseq                    | Pretraining                 | Fairseq                                 | Custom: Transformer encoder-decoder (12 encoder / 6 decoder layers) |
