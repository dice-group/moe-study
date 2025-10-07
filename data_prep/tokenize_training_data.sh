#!/bin/bash
#SBATCH --job-name=tokenize_all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-999%100
#SBATCH --output=logs/sharded_tok/tokenize_%A_%a.out
#SBATCH -p normal

PARENT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples
SHARDED_DIR=$PARENT_DIR/samples_sharded
OUT_DIR=/scratch/hpc-prf-merlin/project_data/moe_study/tokenized
TOKENIZER=/scratch/hpc-prf-merlin/project_data/moe_study/250K_vocab_size/trained_multilingual_tokenizer_llama-3.2-1B

SMALL_FILES=$(find $PARENT_DIR -mindepth 2 -maxdepth 2 -type f -name "*.jsonl.gz" \
  | grep -Ev "english|rus_Cyrl|spa_Latn")
BIG_FILES=$(find $SHARDED_DIR -type f -name "part_*.gz")
FILES=($(echo -e "$SMALL_FILES\n$BIG_FILES" | sort))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "Processing file: $FILE"
python tokenize_training_data.py \
    --input-file $FILE \
    --out-dir $OUT_DIR \
    --tokenizer $TOKENIZER \
    --num-proc $SLURM_CPUS_PER_TASK
