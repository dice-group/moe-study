#!/bin/bash
#SBATCH -J adamole_llama3.2-1b
#SBATCH -o logs/adamole/llama3.2-1b_slurm-%j.out
#SBATCH -e logs/adamole/llama3.2-1b_slurm-%j.err
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1

set -e
module reset
module load tools
module load cuda
module load python/3.11

source /opt/software/cluster/EB-SW/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate /scratch/anonymous-project/htyllm-pg/anonymous-user/conda/envs/moe-study

export LOG_LEVEL=DEBUG
export HF_HOME=/scratch/anonymous-project/shared_cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=/scratch/anonymous-project/shared_cache/huggingface/datasets
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting training job"
srun python src/train_hf.py \
  --config /scratch/anonymous-project/moe-study/configs/train_adamole_hf.yaml \
  --tokenized_dataset_dir /scratch/anonymous-project/project_data/moe_study/tokenized \
  --tokenizer /scratch/anonymous-project/project_data/moe_study/250K_vocab_size/trained_multilingual_tokenizer_llama-3.2-1B \
  --use_method adamole \
  --lora_r 4 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_targets "q_proj,k_proj,v_proj,o_proj" \
  --adamole_num_experts 8 \
  --adamole_max_threshold 0.25 \
  --adamole_aux_coef 0.001 \
  --gradient_checkpointing \
  --do_harness_eval \
  --harness_tasks belebele hellaswag arc xnli \
  --wandb_project moe-study

echo "Job completed"
