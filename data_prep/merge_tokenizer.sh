#!/bin/bash
#SBATCH --job-name=test_merge_tokenizer_llama3.2-1b
#SBATCH --output=logs/merge_tok/test_merge_tokenizers_vocab_llama3.2-1b_%j.out
#SBATCH --error=logs/merge_tok/metest_merge_tokenizers_vocab_llama3.2-1b_%j.err
#SBATCH --array=0-199
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00


LANGS=($(ls /scratch/anonymous-project/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset/*.jsonl.gz | xargs -n1 basename | sed 's/.jsonl.gz//'))
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

srun python -u merge_tokenizer.py \
  --base_tokenizer meta-llama/Llama-3.2-1B \
  --target_tokenizer /scratch/anonymous-project/project_data/moe_study/trained_multilingual_tokenizer_llama-3.2-1B \
  --data_dir /scratch/anonymous-project/project_data/moe_study/fw_samples/preprocessed_tokenizer_subset \
  --min_freq 5 \
  --limit 100000 \
  --output_dir /scratch/anonymous-project/project_data/moe_study/fw_samples/extended_tokenizer/test \
  --lang ${LANG} \