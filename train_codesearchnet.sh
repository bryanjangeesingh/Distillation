#!/bin/bash

# Training script for CodeSearchNet dataset with CodeLlama student and WizardCoder teacher

# Set environment variables for better performance
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export TOKENIZERS_PARALLELISM=true
export BITSANDBYTES_USE_4BIT=True
export LD_LIBRARY_PATH=/home/dornera/miniconda3/lib:$LD_LIBRARY_PATH
# Run the training script with optimal parameters
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 \
    finetuning.py \
    --model_name "codellama/CodeLlama-7b-hf" \
    --dataset.file "datasets/loader/load.py" \
    --lr 5e-5 \
    --num_epochs 3 \
    --batch_size_training 16 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir "train/output/codellama_codesearchnet" \
    --distillation \
    --distillation_config.model_name "WizardLMTeam/WizardCoder-Python-13B-V1.0" \
    --distillation_config.enable_fsdp \
    --distillation_config.pure_bf16 \
    --distillation_config.distil_factor 1.0 \
    --distillation_config.teacher_temperature 2.0 \
    --distillation_config.student_temperature 2.0 \
    --save_step 500 \
    --use_peft \
    --peft_method "lora" \
    --enable_fsdp \
    --mixed_precision \
    --use_fp16 \
    --run_validation False
