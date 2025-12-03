#!/bin/bash

NAME=llava-fusefisrtv3modifiedv4-2nd
#model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune
model_path=/group/30106/aidenqian/TokenPacker-main/checkpoints/llava-fusefirstv3-modified-v4-finetune-bs12-2nd


python -m llava.eval.model_vqa \
    --model-path ${model_path} \
    --question-file /group/30106/aidenqian/TokenPacker-main/playground/data/eval/mm-vet/llava-mm-vet-processed.jsonl \
    --image-folder /group/30106/aidenqian/VLMEvalKit/eval_data/images/MMVet \
    --answers-file ./playground/data/eval/mm-vet/answers/${NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${NAME}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${NAME}.json
