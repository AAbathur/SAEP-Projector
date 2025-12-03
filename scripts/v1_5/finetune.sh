#!/bin/bash


model_path=/Path/of/Vicuna-1.5-7B
vision_tower=/Path/of/CLIP-ViT-Large-Patch14-336
projector_type=SAEProjector ## set the projector name, includes: SAEProjector, resampler, ldpv2, cabstractor, tokenpacker, 
select_multi_layer='[9,13,17,20,22]' ## set the visual feature layer index, start from 0
pretrain_mm_mlp_adapter=./stage1-pretrain/output_dir/mm_projector.bin ## stage1 pre-trained projector weight path
batch_size=12 
gradient_accumulation_steps=1
output_dir=./stage2-tuning/output_dir




CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${model_path} \
    --version llava_v1 \
    --data_path ./stage2/data/path \
    --image_folder ./stage2/image/path \
    --vision_tower ${vision_tower} \
    --pretrain_mm_mlp_adapter ${pretrain_mm_mlp_adapter} \
    --mm_projector_type ${projector_type} \
    --scale_factor 2 \
    --mm_vision_select_layer -2 \
    --select_multi_layer ${select_multi_layer} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --ddp_backend "nccl" \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "none"
