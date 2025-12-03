#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


#CKPT=llava-fusefistv3-modifiedv4
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune

CKPT=llava-ldpv2
model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-ldpv2-finetune

#CKPT=llava-cabstractor
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-cabstractor-finetune

#CKPT=llava-resampler
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-resampler144-910b


#CKPT="llava-tokenpacker-7b"
## SPLIT="llava_vqav2_mscoco_test-dev2015" ## 107394
SPLIT="llava_vqav2_mscoco_test-dev2015" ## 447793

<<"COMMENT"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${model_path} \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /group/30106/public/vqa2/test2015/ \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

COMMENT

#<<"COMMENT"
wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

#COMMENT
