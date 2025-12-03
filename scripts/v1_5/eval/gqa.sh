#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

#CKPT="llava-tokenpacker-7b-910b-v2"
#GQADIR="./playground/data/eval/gqa/tokenpacker-910b-v2"
#model_path=checkpoints/llava-tokenpacker-910b-v2

#CKPT="llava-fusefistv3-910b"
#GQADIR="./playground/data/eval/gqa/llava-fusefisrtv3-910b"
#model_path=checkpoints/llava-fusefistv3-910b

#CKPT="llava-groupedfusefisrtv8-910b"
#CKPT="llava-tokenpacker-7b-910b-bs16"
#GQADIR="./playground/data/eval/gqa/llava-tokenpacker-7b-910b-bs16"
#model_path=checkpoints/llava-tokenpacker-910b-bs16

#CKPT="llava-fusefisrtv3-a100"
#GQADIR="./playground/data/eval/gqa/llava-fusefisrtv3-a100"
#model_path=checkpoints/llava-fusefisrtv3-a100
#CKPT="llava-groudedfusefirstv8-a100"
#GQADIR="./playground/data/eval/gqa/llava-groudedfusefirstv8-a100"
#model_path=checkpoints/llava-groudedfusefirstv8-a100

#CKPT="llava-resampler-910b"
#GQADIR="./playground/data/eval/gqa/llava-resampler-910b"
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-resampler144-910b

#CKPT="llava-fusefirstv3-modified-v4"
#GQADIR="./playground/data/eval/gqa/llava-fusefirstv3-modified-v4"
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune

#CKPT=llava-tokenpacker
#GQADIR=./playground/data/eval/gqa/llava-tokenpacker
#model_path=/group/30106/aidenqian/PublicModels/Tokenpacker-7b-144

#model=llava-cabstractor
#GQADIR=./playground/data/eval/gqa/llava-cabstractor
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-cabstractor-finetune

#model=llava-ldpv2
#GQADIR=./playground/data/eval/gqa/llava-ldpv2
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-ldpv2-finetune

model=llava-13b-fusefisrtv3modifiedv4
GQADIR=./playground/data/eval/gqa/llava-13b-fusefisrtv3modifiedv4
model_path=checkpoints/llava-13b-fusefirstv3-modified-v4-finetune

SPLIT="llava_gqa_testdev_balanced"
gqa_images="/group/30106/aidenqian/prismatic-vlms-main/data/download/llava-v1.5-instruct/gqa/images"

#<<"COMMENT"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${model_path} \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ${gqa_images} \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

#COMMENT

#<<"COMMENT"
wait


output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir $GQADIR

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cp ./playground/data/eval/gqa/testdev_balanced_questions.json $GQADIR/testdev_balanced_questions.json ## 复制一个question文件过去，计算score需要用

cd $GQADIR
python /group/30106/aidenqian/TokenPacker-main/playground/data/eval/gqa/eval.py --tier testdev_balanced

#rm ./testdev_balanced_questions.json

#COMMENT