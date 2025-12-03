#!/bin/bash

### prismatic-vlm中的评估是pope最终准确率是各项均值
## 

#NAME="llava-tokenpacker-7b"
#model_path=/group/30106/aidenqian/PublicModels/Tokenpacker-7b-144
#NAME="llava-tokenpacker-7b-a100"
#model_path=checkpoints/llava-tokenpacker

#NAME="llava-tokenpacker-7b-910b"
#model_path=checkpoints/llava-tokenpacker-910b
#NAME="llava-tokenpacker-910b-v2"
#model_path=checkpoints/llava-tokenpacker-910b-v2
#NAME="llava-fusefistv3-910b"
#model_path=checkpoints/llava-fusefistv3-910b

#NAME="llava-tokenpacker-7b-910b-bs16"
#model_path=checkpoints/llava-tokenpacker-910b-bs16

#NAME="llava-groudedfusefirstv8-a100"
#model_path=checkpoints/llava-groudedfusefirstv8-a100
#NAME="llava-fusefirstv3-a100"
#model_path=checkpoints/llava-fusefisrtv3-a100

NAME=llava-13b-fusefisrtv3modifiedv4
model_path=checkpoints/llava-13b-fusefirstv3-modified-v4-finetune
#model_path=checkpoints/llava-fusefirstv3-modified-v4-noinit-finetune-bs12

#NAME=llava-fusefisrtv3modifiedv4
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune

python -m llava.eval.model_vqa_loader_pope \
    --model-path ${model_path} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /group/30106/aidenqian/down_stream_datasets/coco_image/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/annotations \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$NAME.jsonl

