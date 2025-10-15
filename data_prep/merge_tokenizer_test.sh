LANG="hin_Deva"

python -u merge_tokenizer.py \
  --base_tokenizer meta-llama/Llama-3.1-8B \
  --target_tokenizer /scratch/hpc-prf-merlin/project_data/moe_study/trained_multilingual_tokenizer_llama-3.1-8B \
  --data_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
  --output_dir /scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/extended_tokenizer/test_one \
  --min_freq 5 \
  --limit 5000 \
  --lang $LANG
