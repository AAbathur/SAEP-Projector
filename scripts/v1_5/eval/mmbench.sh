#!/bin/bash

SPLIT="mmbench_dev_20230712"

model_id=llava-fusefistv3-modifiedv4
model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune

python -m llava.eval.model_vqa_mmbench \
    --model-path ${model_path} \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${model_id}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${model_id}