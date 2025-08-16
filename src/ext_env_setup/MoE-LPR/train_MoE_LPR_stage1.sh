cd MoE-LPR
ROOT_DIR="$(pwd)/LLaMA-Factory" # this will be /upb/users/j/joeldag/profiles/unix/cs/MERLIN/moe-study/src/ext_env_setup/MoE-LPR/MoE-LPR
echo -e "Root dir $ROOT_DIR"
MODEL_PATH=meta-llama/Meta-Llama-3-8B
OUTPUT_DIR=./output_dir
DATASET=wiki_demo

export WANDB_DISABLED=true
deepspeed --num_gpus 1 --master_port=9902 LLaMA-Factory/src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $MODEL_PATH \
    --finetuning_type moe \
    --topk 1 \6
    --moe_num_experts 2 \
    --aux_loss_coef 0.01 \
    --do_train \
    --dataset_dir $ROOT_DIR/data \
    --dataset $DATASET \
    --preprocessing_num_workers 16 \
    --cutoff_len 1024 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 10 \
    --save_steps 1000 \
    --save_only_model \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16